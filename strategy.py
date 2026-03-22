#!/usr/bin/env python3
"""
Experiment #017: 12h KAMA-ADX Trend + BB Squeeze Breakout + 1D/1W HTF Filter

Hypothesis: After 16 failed experiments, the pattern shows:
1. CRSI mean-reversion (#011) failed because it fights strong trends
2. Lower TFs (15m-1h) suffer from noise and fee drag
3. 12h timeframe needs fewer, higher-quality signals (20-40 trades/year)
4. KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA
5. BB Squeeze + ADX combo catches explosive breakouts with confirmation
6. Asymmetric exits (harder to exit in trend) improve win rate

This 12h strategy combines:

1. KAMA(10,2,30): Adaptive MA that smooths in chop, follows in trends.
   More robust than HMA for crypto's regime changes.

2. ADX(14) + DI+/DI-: Trend strength filter. ADX>25 = trend, ADX<20 = range.
   Only enter breakouts when ADX rising (confirming momentum).

3. Bollinger Band Squeeze: BB Width < 20th percentile = compression.
   Breakout from squeeze + ADX confirmation = high-probability move.

4. 1D HMA trend bias: Medium-term HTF filter (more responsive than 1W).
   Long only if price > 1D_HMA, short only if price < 1D_HMA.

5. 1W HMA regime: Ultra-long-term bias. Avoid shorts if price >> 1W_HMA.

6. Volume confirmation: Entry volume > 1.5x 20-bar avg = institutional interest.

7. Asymmetric exits: In trending regime (ADX>30), hold through pullbacks.
   In range regime (ADX<20), take profit at BB opposite band.

8. ATR(14) trailing stop: 2.5*ATR for longs, 2.5*ATR for shorts.

Why this should beat #011 (Sharpe=-0.271):
- KAMA adapts better than HMA to crypto's changing volatility
- BB Squeeze + ADX catches explosive moves, not mean-reversion fights
- 1D HMA more responsive than 1W for 12h entries
- Volume filter reduces false breakouts
- Asymmetric exits improve win rate in trends
- Target 25-40 trades/year on 12h (optimal frequency, low fee drag)

Timeframe: 12h (REQUIRED)
HTF: 1D and 1W via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete, regime-adaptive
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_bb_squeeze_1d_1w_hma_vol_asym_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    Smoothing Constant (SC) = (ER * (fast_SC - slow_SC) + slow_SC)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change over ER period
    change = np.abs(close_s - close_s.shift(er_period)).values
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
    
    # Efficiency Ratio (0 = noise, 1 = perfect trend)
    er = np.zeros(n)
    for i in range(er_period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index) + DI+/DI-.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    n = len(close)
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_s[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / tr_s[i]
    
    # DX = |DI+ - DI-| / (DI+ + DI-)
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = SMA of DX
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx, plus_di, minus_di

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    
    # Middle band (SMA)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    
    # Standard deviation
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    # Upper and lower bands
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    # Bandwidth (normalized)
    bandwidth = np.zeros(len(close))
    for i in range(period, len(close)):
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bandwidth[i] = 0
    
    # Percentile rank of bandwidth (for squeeze detection)
    bw_percentile = np.zeros(len(close))
    for i in range(period * 2, len(close)):
        bw_window = bandwidth[max(0, i-period*2):i+1]
        bw_percentile[i] = np.sum(bw_window <= bandwidth[i]) / len(bw_window) * 100
    
    return upper, middle, lower, bandwidth, bw_percentile

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    bb_upper, bb_middle, bb_lower, bb_bw, bb_bw_pct = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_TREND = 0.35  # Larger in strong trend
    BASE_SIZE_RANGE = 0.20  # Smaller in choppy/range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    entry_adx = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(kama[i]) or np.isnan(adx[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_bw_pct[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        # === HTF TREND BIAS ===
        # 1D HMA: Medium-term trend direction
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        
        # 1W HMA: Long-term regime (avoid shorts in strong bull)
        bull_1w = close[i] > hma_1w_aligned[i]
        
        # === ADX TREND STRENGTH ===
        is_trending = adx[i] > 25
        is_strong_trend = adx[i] > 30
        is_ranging = adx[i] < 20
        
        # DI+ vs DI- direction
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # ADX rising (momentum confirming)
        adx_rising = adx[i] > adx[i-1] if i > 0 else False
        
        # === BOLLINGER BAND SQUEEZE ===
        # Squeeze: bandwidth in bottom 20% of recent range
        is_squeeze = bb_bw_pct[i] < 20
        
        # Price position relative to BB
        price_near_upper = close[i] > bb_upper[i] * 0.995
        price_near_lower = close[i] < bb_lower[i] * 1.005
        price_above_middle = close[i] > bb_middle[i]
        price_below_middle = close[i] < bb_middle[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmation = volume[i] > 1.5 * vol_sma[i]
        
        # === POSITION SIZING BASED ON REGIME ===
        if is_strong_trend:
            base_size = BASE_SIZE_TREND
        elif is_ranging:
            base_size = BASE_SIZE_RANGE
        else:
            base_size = (BASE_SIZE_TREND + BASE_SIZE_RANGE) / 2
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: BB SQUEEZE BREAKOUT (highest conviction)
        # Requires: squeeze + ADX rising + HTF bias + volume
        if is_squeeze and adx_rising:
            # Long breakout: squeeze + price breaks upper BB + bull bias
            if price_near_upper and bull_1d and di_bull and vol_confirmation:
                new_signal = base_size
            
            # Short breakout: squeeze + price breaks lower BB + bear bias
            elif price_near_lower and bear_1d and di_bear and vol_confirmation:
                # Only short if not in extreme bull regime (1W filter)
                if not bull_1w or adx[i] > 35:  # Allow short in very strong downtrend
                    new_signal = -base_size
        
        # MODE 2: TREND FOLLOWING (no squeeze, but strong ADX)
        elif is_trending and not is_squeeze:
            # Long: strong trend + bull bias + DI+ dominant
            if is_strong_trend and bull_1d and di_bull and price_above_middle:
                new_signal = base_size
            
            # Short: strong trend + bear bias + DI- dominant
            elif is_strong_trend and bear_1d and di_bear and price_below_middle:
                # Only short if 1W also bearish or ADX very strong
                if not bull_1w or adx[i] > 40:
                    new_signal = -base_size
        
        # MODE 3: RANGE MEAN REVERSION (low ADX, BB extremes)
        elif is_ranging:
            # Long: price at lower BB + bull 1D bias (counter-trend in range)
            if price_near_lower and bull_1d:
                new_signal = base_size * 0.7  # Smaller size for mean reversion
            
            # Short: price at upper BB + bear 1D bias
            elif price_near_upper and bear_1d:
                new_signal = -base_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === ASYMMETRIC EXIT LOGIC ===
        # In strong trends (ADX>30), hold through pullbacks unless stoploss hit
        # In ranges (ADX<20), exit at opposite BB band (take profit)
        asymmetric_exit = False
        
        if in_position and position_side != 0 and not is_strong_trend:
            # Range regime: take profit at opposite band
            if position_side > 0 and price_near_upper:
                asymmetric_exit = True  # Long reached upper BB in range
            if position_side < 0 and price_near_lower:
                asymmetric_exit = True  # Short reached lower BB in range
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias flips against position
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and bear_1d and adx[i] > 25:
                trend_reversal = True  # Long but 1D turned bear with trend
            if position_side < 0 and bull_1d and adx[i] > 25:
                trend_reversal = True  # Short but 1D turned bull with trend
        
        # Apply stoploss or exits
        if stoploss_triggered or asymmetric_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                entry_adx = adx[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                entry_adx = adx[i]
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                entry_adx = 0.0
        
        signals[i] = new_signal
    
    return signals