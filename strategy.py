#!/usr/bin/env python3
"""
Experiment #486: 12h Primary + 1d HTF — KAMA Trend + Fisher Transform Reversals

Hypothesis: After 485 failed experiments, clear pattern emerges:
1. Pure trend-following fails on BTC/ETH in bear markets (2022 crash, 2025 range)
2. Fisher Transform catches reversals better than RSI in bear/range regimes
3. KAMA (Kaufman Adaptive MA) reduces whipsaw vs HMA/EMA in choppy markets
4. 12h timeframe balances trade frequency (20-50/year) with fee drag
5. Volume confirmation prevents false breakouts (taker_buy_volume ratio)

Why this might beat current best (Sharpe=0.435):
- Fisher Transform proven in research for bear market reversals
- KAMA adapts to volatility, less whipsaw than fixed-period MAs
- 1d KAMA provides cleaner trend filter than HMA
- Relaxed Fisher thresholds (±1.5) ensure adequate trade frequency
- Asymmetric sizing (0.30 long, 0.25 short) protects in bear markets

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-60 trades on train, 3+ on test, EACH symbol Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise: fast in trends, slow in chop.
    ER (Efficiency Ratio) = |change| / sum(|changes|)
    SC (Smoothing Constant) = ER * (fast - slow) + slow
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    change = np.abs(close_s.diff(er_period).values)
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing Constant
    fast = 2.0 / (fast_sc + 1)
    slow = 2.0 / (slow_sc + 1)
    sc = er * (fast - slow) + slow
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        range_hl = hh - ll
        if range_hl == 0:
            range_hl = 1e-10
        
        # Normalized price
        price_norm = 0.66 * ((high[i] + low[i]) / 2.0 - ll) / range_hl + 0.67
        
        # Clamp to (0, 1)
        price_norm = np.clip(price_norm, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm))
        
        # Previous fisher for signal
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF KAMA (major trend direction)
    kama_1d_50 = calculate_kama(df_1d['close'].values, er_period=10, fast_sc=2, slow_sc=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_50_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_21 = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=21)
    kama_12h_50 = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=50)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.zeros(n)
    mask = volume > 0
    vol_ratio[mask] = taker_buy_vol[mask] / volume[mask]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Fisher crosses to avoid repeated signals
    prev_fisher_long_signal = False
    prev_fisher_short_signal = False
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1d_50_aligned[i]):
            continue
        if np.isnan(kama_12h_21[i]) or np.isnan(kama_12h_50[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (KAMA direction) ===
        bull_regime = close[i] > kama_1d_50_aligned[i]
        bear_regime = close[i] < kama_1d_50_aligned[i]
        
        # === 12H LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_12h_21[i] > kama_12h_50[i]
        kama_bearish = kama_12h_21[i] < kama_12h_50[i]
        
        # === KAMA SLOPE (momentum confirmation) ===
        kama_slope_long = kama_12h_21[i] > kama_12h_21[i-1] if i > 0 else False
        kama_slope_short = kama_12h_21[i] < kama_12h_21[i-1] if i > 0 else False
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === VOLUME CONFIRMATION ===
        vol_buy_pressure = vol_ratio[i] > 0.55  # More buying volume
        vol_sell_pressure = vol_ratio[i] < 0.45  # More selling volume
        
        # === RSI FILTER (avoid extreme counter-trend) ===
        rsi_not_extreme_long = rsi_14[i] < 80  # Don't long at extreme overbought
        rsi_not_extreme_short = rsi_14[i] > 20  # Don't short at extreme oversold
        
        # === ENTRY LOGIC — RELAXED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions, any can trigger)
        if fisher_long_cross and kama_bullish and rsi_not_extreme_long:
            new_signal = LONG_SIZE
        elif bull_regime and fisher[i] < -1.0 and kama_slope_long:
            new_signal = LONG_SIZE
        elif kama_bullish and fisher[i] < -0.5 and vol_buy_pressure:
            new_signal = LONG_SIZE * 0.8
        elif bull_regime and kama_bullish and rsi_14[i] < 40:
            new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (multiple conditions, any can trigger)
        if new_signal == 0.0:
            if fisher_short_cross and kama_bearish and rsi_not_extreme_short:
                new_signal = -SHORT_SIZE
            elif bear_regime and fisher[i] > 1.0 and kama_slope_short:
                new_signal = -SHORT_SIZE
            elif kama_bearish and fisher[i] > 0.5 and vol_sell_pressure:
                new_signal = -SHORT_SIZE * 0.8
            elif bear_regime and kama_bearish and rsi_14[i] > 60:
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
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long when Fisher overbought
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        # Exit short when Fisher oversold
        if in_position and position_side < 0 and fisher[i] < -1.5:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and bear_regime and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and kama_bullish:
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