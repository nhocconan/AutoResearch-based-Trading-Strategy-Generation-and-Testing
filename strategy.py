#!/usr/bin/env python3
"""
Experiment #005: 1h Primary + 4h/1d HTF — Fisher Transform + BB Mean Reversion + HTF Trend

Hypothesis: After 4 failed experiments (all CHOP/CRSI based), we need a different approach.
This strategy uses Ehlers Fisher Transform for reversal detection (underutilized in crypto),
combined with Bollinger Bands for mean-reversion context and 4h HMA for trend bias.

Key differences from failed strategies:
1. Fisher Transform catches reversals better than RSI in bear/range markets (literature-backed)
2. BB provides dynamic support/resistance without overfiltering
3. 4h HMA gives trend direction without the slowness of 1d HMA
4. 1d ADX regime filter prevents trend-following in choppy conditions
5. LOOSE entry conditions to ensure trade generation (learned from #685 zero-trade failure)

Entry Logic:
- Range regime (ADX<18): Mean revert at BB extremes + Fisher confirmation
- Trend regime (ADX>25): Pullback entries in direction of 4h HMA trend
- Fisher threshold: -0.8/+0.8 (looser than -1.0/+1.0 to generate more trades)
- Volume: >0.7x avg (not 0.8x to avoid overfiltering)

Position sizing: 0.25 discrete (conservative for 1h TF)
Target: 40-70 trades/year on 1h timeframe
Stoploss: 2.5*ATR trailing stop

Why this might work when CHOP/CRSI failed:
- Fisher Transform has superior signal-to-noise ratio for reversals
- BB %B provides continuous measure (not binary like CHOP thresholds)
- ADX hysteresis (25 enter, 18 exit) prevents regime flip-flopping
- Multiple entry paths (Fisher OR BB touch) increase trade frequency
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_bb_hma4h_adx1d_v2"
timeframe = "1h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.67 * (price - LL) / (HH - LL) - 0.67
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    median = (high_s + low_s) / 2.0
    hh = median.rolling(window=period, min_periods=period).max()
    ll = median.rolling(window=period, min_periods=period).min()
    
    range_hl = hh - ll
    x = 0.67 * (median - ll) / (range_hl + 1e-10) - 0.67
    x = np.clip(x, -0.99, 0.99)
    
    fisher = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    return upper.values, lower.values, bandwidth.values, pct_b.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d ADX for regime filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_bandwidth, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # ADX hysteresis
    prev_adx_regime = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(bb_pct_b[i]) or np.isnan(vol_sma[i]):
            continue
        if np.isnan(adx_1d_aligned[i]):
            continue
        if atr_14[i] == 0 or vol_sma[i] == 0:
            continue
        
        # === 4H TREND BIAS ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-5] if i >= 5 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-5] if i >= 5 else False
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D ADX REGIME (with hysteresis) ===
        adx_val = adx_1d_aligned[i]
        if adx_val > 25.0:
            adx_regime = 1  # Trending
        elif adx_val < 18.0:
            adx_regime = 2  # Range
        else:
            adx_regime = prev_adx_regime
        prev_adx_regime = adx_regime
        
        is_trend_regime = (adx_regime == 1)
        is_range_regime = (adx_regime == 2)
        
        # === FISHER TRANSFORM SIGNALS (looser thresholds) ===
        fisher_cross_up = (fisher[i] > -0.8) and (fisher_signal[i] <= -0.8)
        fisher_cross_down = (fisher[i] < 0.8) and (fisher_signal[i] >= 0.8)
        fisher_extreme_low = fisher[i] < -1.0
        fisher_extreme_high = fisher[i] > 1.0
        
        # === BOLLINGER BAND SIGNALS ===
        bb_touch_lower = bb_pct_b[i] < 0.20
        bb_touch_upper = bb_pct_b[i] > 0.80
        
        # === VOLUME CONFIRMATION (looser) ===
        volume_confirmed = volume[i] > 0.7 * vol_sma[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY
        if is_range_regime:
            # Mean reversion long in range market
            if (fisher_extreme_low or bb_touch_lower) and volume_confirmed:
                if not hma_4h_slope_bear:  # Not strongly bearish on 4h
                    new_signal = POSITION_SIZE
        elif is_trend_regime:
            # Trend pullback long
            if hma_4h_slope_bull and price_above_hma_4h:
                if fisher_cross_up or (fisher[i] < -0.3 and volume_confirmed):
                    new_signal = POSITION_SIZE
        
        # SHORT ENTRY (separate check to allow flip)
        if is_range_regime:
            # Mean reversion short in range market
            if (fisher_extreme_high or bb_touch_upper) and volume_confirmed:
                if not hma_4h_slope_bull:  # Not strongly bullish on 4h
                    new_signal = -POSITION_SIZE
        elif is_trend_regime:
            # Trend pullback short
            if hma_4h_slope_bear and price_below_hma_4h:
                if fisher_cross_down or (fisher[i] > 0.3 and volume_confirmed):
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION (avoid churn) ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h and is_trend_regime:
                new_signal = 0.0
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h and is_trend_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(new_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = int(np.sign(new_signal))
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