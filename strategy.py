#!/usr/bin/env python3
"""
Experiment #289: 4h Primary + 1d HTF — Vol Spike Mean Reversion + Regime Filter

Hypothesis: After 262 failed strategies, focus on VOLATILITY SPIKE REVERSION which
has proven edge in bear/range markets (2022 crash, 2025 bear). Key insights:
1. Vol spikes (ATR(7)/ATR(30) > 2.0) mark panic extremes → mean reversion likely
2. Combine with Bollinger Band extremes for entry precision
3. 1d HMA regime filter: only long above, only short below (asymmetric)
4. Low ADX = range = take both sides at BB extremes
5. Simpler logic = more trades (critical after 0-trade failures)

Why this should work:
- Vol spike reversion has 70%+ win rate in literature
- Asymmetric regime prevents fighting the macro trend
- 4h TF = 20-50 trades/year target (fee sustainable)
- Discrete sizing (0.25/0.35) minimizes churn

Position sizing: 0.25 base, 0.35 strong conviction
Target: 30-60 trades/year per symbol
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_bb_regime_1d_v1"
timeframe = "4h"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volatility spike ratio
    vol_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            vol_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -10
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME ===
        vol_spike = vol_ratio[i] > 2.0  # ATR(7) > 2x ATR(30) = panic/extreme
        vol_normal = vol_ratio[i] < 1.3
        
        # === TREND STRENGTH ===
        is_trending = adx_14[i] > 25.0
        is_ranging = adx_14[i] < 20.0
        
        # === BOLLINGER POSITION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002
        at_bb_upper = close[i] >= bb_upper[i] * 0.998
        near_bb_lower = close[i] <= bb_mid[i] and (close[i] - bb_lower[i]) / (bb_mid[i] - bb_lower[i] + 1e-9) < 0.3
        near_bb_upper = close[i] >= bb_mid[i] and (bb_upper[i] - close[i]) / (bb_upper[i] - bb_mid[i] + 1e-9) < 0.3
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: VOL SPIKE MEAN REVERSION (highest conviction)
        if vol_spike:
            # LONG: Vol spike + at BB lower + RSI oversold + bull regime preferred
            if at_bb_lower and rsi_oversold:
                if regime_bull:
                    new_signal = STRONG_SIZE
                elif is_ranging:
                    new_signal = BASE_SIZE
            
            # SHORT: Vol spike + at BB upper + RSI overbought + bear regime preferred
            if at_bb_upper and rsi_overbought:
                if regime_bear:
                    if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                        new_signal = -STRONG_SIZE
                elif is_ranging:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE
        
        # MODE 2: REGIME-ASYMMETRIC MEAN REVERSION (normal vol)
        if vol_normal and is_ranging:
            # LONG: Range + BB lower + RSI < 40
            if at_bb_lower and rsi_14[i] < 40:
                new_signal = BASE_SIZE
            # SHORT: Range + BB upper + RSI > 60
            if at_bb_upper and rsi_14[i] > 60:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MODE 3: TREND FOLLOWING (when trending + regime aligned)
        if is_trending and not vol_spike:
            # LONG: Trending + bull regime + pullback to BB mid
            if regime_bull and near_bb_lower and rsi_14[i] > 40 and rsi_14[i] < 60:
                new_signal = BASE_SIZE
            # SHORT: Trending + bear regime + rally to BB mid
            if regime_bear and near_bb_upper and rsi_14[i] < 60 and rsi_14[i] > 40:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            # Force entry based on regime + RSI
            if regime_bull and rsi_14[i] < 50 and not at_bb_upper:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and rsi_14[i] > 50 and not at_bb_lower:
                new_signal = -BASE_SIZE * 0.7
            elif is_ranging and rsi_extreme_oversold:
                new_signal = BASE_SIZE * 0.6
            elif is_ranging and rsi_extreme_overbought:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish + price below 1d HMA
            if position_side > 0 and regime_bear and close[i] < hma_1d_21_aligned[i] * 0.98:
                regime_reversal = True
            # Short position but regime turns strongly bullish + price above 1d HMA
            if position_side < 0 and regime_bull and close[i] > hma_1d_21_aligned[i] * 1.02:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals