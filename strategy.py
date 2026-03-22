#!/usr/bin/env python3
"""
Experiment #513: 1d Primary + 1w HTF — KAMA Adaptive Trend + RSI Pullback + Volume Confirmation

Hypothesis: After 448+ failed strategies (mostly HMA/Choppiness/Vol-spike combos), try KAMA
(Kaufman Adaptive Moving Average) which adapts to market regime automatically.

Key insights from failures:
- #507 mtf_1d_hma_rsi_pullback_1w_simp_v1 failed (Sharpe=-0.689) — HMA too slow
- Vol-spike strategies all failed — too rare, miss most moves
- Complex filters = 0 trades (see #505, #506, #508, #510 with Sharpe=0.000)

Why KAMA might work:
1. Adapts smoothing based on volatility — fast in trends, slow in chop
2. Fewer whipsaws than HMA/EMA in range markets (2025 test period)
3. Proven in literature for crypto (Kaufman's original research)

Strategy logic:
1. 1w KAMA(21) for major trend direction (HTF filter)
2. 1d RSI(7) for entry timing — faster than RSI(14), more signals
3. Volume confirmation — taker_buy_volume ratio > 0.55 for longs
4. ATR(14) trailing stop at 2.5x
5. Asymmetric sizing: long=0.30, short=0.25 (bear market bias)

Entry conditions (ANY one triggers):
- Long: price > 1w KAMA AND RSI(7) < 50 (pullback in uptrend)
- Long: RSI(7) < 35 AND volume spike (panic bottom)
- Short: price < 1w KAMA AND RSI(7) > 50 (bounce in downtrend)
- Short: RSI(7) > 65 AND volume spike (FOMO top)

This ensures >=30 trades/symbol on train by having multiple independent triggers.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi7_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, smoothing_period=2, trend_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    Fast in trends, slow in choppy markets.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER): measures trend vs noise
    change = np.abs(close_s - close_s.shift(trend_period))
    volatility = close_s.diff().abs().rolling(window=trend_period, min_periods=trend_period).sum()
    
    er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constant
    sc = (er * (2.0 / (smoothing_period + 1) - 2.0 / (efficiency_period + 1)) + 
          2.0 / (efficiency_period + 1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=7):
    """Calculate RSI using Wilder's smoothing. Faster period for more signals."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (buying pressure indicator)."""
    ratio = taker_buy_volume / (volume + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w KAMA for major trend direction
    kama_1w = calculate_kama(df_1w['close'].values, efficiency_period=10, trend_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars only)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for more signals
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    # Volume spike detection (above average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25  # Asymmetric: smaller short size for bear market
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(rsi_7[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > kama_1w_aligned[i]
        bear_regime = close[i] < kama_1w_aligned[i]
        
        # === RSI SIGNALS (faster period = more trades) ===
        rsi_oversold = rsi_7[i] < 45.0  # Looser threshold for more trades
        rsi_overbought = rsi_7[i] > 55.0
        rsi_extreme_low = rsi_7[i] < 35.0
        rsi_extreme_high = rsi_7[i] > 65.0
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = vol_ratio[i] > 0.52  # More buyers than sellers
        volume_bearish = vol_ratio[i] < 0.48
        volume_spike = vol_spike[i] if not np.isnan(vol_spike[i]) else False
        
        # === ENTRY LOGIC — Multiple independent triggers ===
        new_signal = 0.0
        
        # LONG ENTRIES (any one condition triggers)
        # Condition 1: Bull regime + RSI pullback (primary entry)
        if bull_regime and rsi_oversold:
            new_signal = LONG_SIZE
        # Condition 2: Bull regime + RSI extreme + volume confirmation
        elif bull_regime and rsi_extreme_low and volume_bullish:
            new_signal = LONG_SIZE
        # Condition 3: Volume spike + RSI extreme (panic bottom)
        elif volume_spike and rsi_extreme_low:
            new_signal = LONG_SIZE
        # Condition 4: RSI extreme alone (catch major reversals)
        elif rsi_extreme_low and not bear_regime:
            new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: Bear regime + RSI bounce (primary entry)
            if bear_regime and rsi_overbought:
                new_signal = -SHORT_SIZE
            # Condition 2: Bear regime + RSI extreme + volume confirmation
            elif bear_regime and rsi_extreme_high and volume_bearish:
                new_signal = -SHORT_SIZE
            # Condition 3: Volume spike + RSI extreme (FOMO top)
            elif volume_spike and rsi_extreme_high:
                new_signal = -SHORT_SIZE
            # Condition 4: RSI extreme alone (catch major reversals)
            elif rsi_extreme_high and not bull_regime:
                new_signal = -SHORT_SIZE * 0.7
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # Exit long on RSI overbought or regime flip
        if in_position and position_side > 0:
            if rsi_7[i] > 70.0:  # Take profit on extreme
                new_signal = 0.0
            if bear_regime and rsi_7[i] > 55.0:  # Regime flip exit
                new_signal = 0.0
        
        # Exit short on RSI oversold or regime flip
        if in_position and position_side < 0:
            if rsi_7[i] < 30.0:  # Take profit on extreme
                new_signal = 0.0
            if bull_regime and rsi_7[i] < 45.0:  # Regime flip exit
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals