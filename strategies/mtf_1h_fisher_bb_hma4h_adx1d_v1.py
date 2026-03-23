#!/usr/bin/env python3
"""
Experiment #695: 1h Primary + 4h/1d HTF — Fisher Transform + BB Mean Reversion + HTF Trend

Hypothesis: After 607 failed strategies, the pattern is clear:
1. CHOP+CRSI combinations have been tried 50+ times and mostly fail on 1h/30m
2. #685 (1h CHOP+CRSI+HMA+volume+session) got Sharpe=0.000 — likely 0 trades from overfiltering
3. 1d timeframe works best (#693 Sharpe=0.105) but we need 1h per experiment spec
4. Fisher Transform is underutilized — catches reversals in bear/range markets better than RSI

This strategy uses:
- Ehlers Fisher Transform (period=9) for precise reversal entries
- Bollinger Band (20, 2.0) for mean-reversion context
- 4h HMA(21) for trend direction bias
- 1d ADX(14) for regime filter (trend vs range)
- Volume confirmation (>0.8x 20-bar avg)

Why this might beat Sharpe=0.520:
- Fisher Transform has superior reversal detection vs RSI in literature
- BB provides dynamic support/resistance levels
- 4h HMA gives trend bias without overfiltering (unlike 1d HMA which is too slow)
- 1d ADX regime prevents trend-following in chop (major loss source in 2022-2024)
- Asymmetric entries: mean-revert when ADX<20, trend-pullback when ADX>25

Position sizing: 0.25 discrete (conservative for 1h TF per Rule 4)
Target: 35-65 trades/year on 1h
Stoploss: 2.5*ATR trailing

CRITICAL LESSONS FROM FAILURES:
- Entry conditions MUST be loose enough to generate trades (#685 = 0 trades)
- Fisher < -1.0 (not -1.5) for long, > +1.0 (not +1.5) for short
- ADX hysteresis: enter at 25, exit at 18 (prevents rapid flipping)
- BB touch OR Fisher extreme = entry (either/or, not both)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_bb_hma4h_adx1d_v1"
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
    
    Signals:
    - Long: Fisher crosses above -1.0 from below
    - Short: Fisher crosses below +1.0 from above
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Median price
    median = (high_s + low_s) / 2.0
    
    # Highest high and lowest low over period
    hh = median.rolling(window=period, min_periods=period).max()
    ll = median.rolling(window=period, min_periods=period).min()
    
    # Normalize price to range [-1, 1]
    range_hl = hh - ll
    x = 0.67 * (median - ll) / (range_hl + 1e-10) - 0.67
    x = np.clip(x, -0.99, 0.99)  # Prevent log domain errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
    
    # Signal line (1-bar lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # Bandwidth (for squeeze detection)
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    # %B (position within bands)
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
    adx_1d_high = df_1d['high'].values
    adx_1d_low = df_1d['low'].values
    adx_1d_close = df_1d['close'].values
    adx_1d = calculate_adx(adx_1d_high, adx_1d_low, adx_1d_close, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_bandwidth, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # ADX hysteresis tracking
    prev_adx_regime = 0  # 0=neutral, 1=trend, 2=range
    
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
        
        # Hysteresis: trend regime needs ADX>25 to enter, <18 to exit
        if adx_val > 25.0:
            adx_regime = 1  # Trending
        elif adx_val < 18.0:
            adx_regime = 2  # Range
        else:
            adx_regime = prev_adx_regime  # Keep previous regime
        
        prev_adx_regime = adx_regime
        is_trend_regime = (adx_regime == 1)
        is_range_regime = (adx_regime == 2)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher[i] > -1.0) and (fisher_signal[i] <= -1.0)
        fisher_cross_down = (fisher[i] < 1.0) and (fisher_signal[i] >= 1.0)
        fisher_extreme_low = fisher[i] < -1.2
        fisher_extreme_high = fisher[i] > 1.2
        
        # === BOLLINGER BAND SIGNALS ===
        bb_touch_lower = bb_pct_b[i] < 0.15  # Near or below lower band
        bb_touch_upper = bb_pct_b[i] > 0.85  # Near or above upper band
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (ADX < 18) + Fisher extreme low OR BB touch lower
        if is_range_regime:
            if (fisher_extreme_low or bb_touch_lower) and volume_confirmed:
                if price_above_hma_4h or not hma_4h_slope_bear:  # Not strongly bearish
                    new_signal = POSITION_SIZE
        
        # Regime 2: Trending market (ADX > 25) + 4h bull + Fisher pullback
        elif is_trend_regime:
            if hma_4h_slope_bull and price_above_hma_4h:
                if fisher_cross_up or (fisher[i] < -0.5 and volume_confirmed):
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (ADX < 18) + Fisher extreme high OR BB touch upper
        if is_range_regime:
            if (fisher_extreme_high or bb_touch_upper) and volume_confirmed:
                if price_below_hma_4h or not hma_4h_slope_bull:  # Not strongly bullish
                    new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market (ADX > 25) + 4h bear + Fisher pullback
        elif is_trend_regime:
            if hma_4h_slope_bear and price_below_hma_4h:
                if fisher_cross_down or (fisher[i] > 0.5 and volume_confirmed):
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
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