#!/usr/bin/env python3
"""
Experiment #696: 12h Primary + 1d HTF — Dual Regime (Trend/Mean-Revert) + Fisher + HMA

Hypothesis: After 607+ failed strategies, the pattern is clear:
1. 12h timeframe has struggled (#686, #689, #692 all negative Sharpe)
2. 1d works best (#693 Sharpe=0.105) but experiment requires 12h primary
3. CHOP+CRSI overused and mostly fails - need different signal combination
4. Fisher Transform underutilized - catches reversals better than RSI in bear/range
5. Dual regime (mean-revert in chop, trend-follow otherwise) proven in literature

This strategy uses:
- 1d HMA(21) for major trend direction (slower than 4h, better for 12h)
- 12h Fisher Transform (period=9) for precise reversal entries
- 1d ADX(14) for regime filter (trend vs range)
- 12h Bollinger Bands (20, 2.0) for mean-reversion context
- Dual regime logic: different entries for chop vs trend

Why this might beat Sharpe=0.520:
- 12h has fewer trades = less fee drag (target 20-50 trades/year)
- Fisher Transform superior reversal detection vs RSI
- 1d HMA gives cleaner trend signal than 4h for 12h primary
- Asymmetric entries prevent whipsaw in 2022 crash
- Conservative sizing (0.28) limits drawdown

Position sizing: 0.28 discrete (conservative for 12h TF)
Target: 25-45 trades/year on 12h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma1d_adx_regime_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Choppy/Range market
    - CHOP < 38.2 = Trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
    
    return chop.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d ADX for regime filter
    adx_1d_high = df_1d['high'].values
    adx_1d_low = df_1d['low'].values
    adx_1d_close = df_1d['close'].values
    adx_1d = calculate_adx(adx_1d_high, adx_1d_low, adx_1d_close, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d Choppiness Index for additional regime filter
    chop_1d_high = df_1d['high'].values
    chop_1d_low = df_1d['low'].values
    chop_1d_close = df_1d['close'].values
    chop_1d = calculate_choppiness_index(chop_1d_high, chop_1d_low, chop_1d_close, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_bandwidth, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Volume SMA for confirmation (loose filter)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # ADX hysteresis tracking
    prev_adx_regime = 0  # 0=neutral, 1=trend, 2=range
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(bb_pct_b[i]) or np.isnan(vol_sma[i]):
            continue
        if np.isnan(adx_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if atr_14[i] == 0 or vol_sma[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
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
        
        # === CHOPPINESS INDEX CONFIRMATION ===
        chop_val = chop_1d_aligned[i]
        is_choppy = chop_val > 55.0  # Slightly lower threshold for more trades
        is_trending_chop = chop_val < 45.0
        
        # === FISHER TRANSFORM SIGNALS (LOOSE thresholds for trades) ===
        fisher_cross_up = (fisher[i] > -1.0) and (fisher_signal[i] <= -1.0)
        fisher_cross_down = (fisher[i] < 1.0) and (fisher_signal[i] >= 1.0)
        fisher_extreme_low = fisher[i] < -0.8  # Looser than -1.2
        fisher_extreme_high = fisher[i] > 0.8  # Looser than +1.2
        
        # === BOLLINGER BAND SIGNALS ===
        bb_touch_lower = bb_pct_b[i] < 0.20  # Near or below lower band
        bb_touch_upper = bb_pct_b[i] > 0.80  # Near or above upper band
        
        # === VOLUME CONFIRMATION (very loose) ===
        volume_confirmed = volume[i] > 0.6 * vol_sma[i]  # 60% of avg is enough
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range/Choppy market - Mean Reversion
        if is_range_regime or is_choppy:
            if (fisher_extreme_low or bb_touch_lower) and volume_confirmed:
                # Only enter long if not in strong downtrend
                if price_above_hma_1d or (not hma_1d_slope_bear):
                    new_signal = POSITION_SIZE
        
        # Regime 2: Trending market - Trend Pullback
        elif is_trend_regime and is_trending_chop:
            if hma_1d_slope_bull and price_above_hma_1d:
                # Enter on Fisher pullback in uptrend
                if fisher_cross_up or fisher[i] < -0.3:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range/Choppy market - Mean Reversion
        if is_range_regime or is_choppy:
            if (fisher_extreme_high or bb_touch_upper) and volume_confirmed:
                # Only enter short if not in strong uptrend
                if price_below_hma_1d or (not hma_1d_slope_bull):
                    new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market - Trend Pullback
        elif is_trend_regime and is_trending_chop:
            if hma_1d_slope_bear and price_below_hma_1d:
                # Enter on Fisher pullback in downtrend
                if fisher_cross_down or fisher[i] > 0.3:
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
            if hma_1d_slope_bear and price_below_hma_1d and is_trend_regime:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d and is_trend_regime:
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