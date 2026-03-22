#!/usr/bin/env python3
"""
Experiment #004: 4h KAMA-ADX Trend with 1d HMA Bias and ATR Risk Management

Hypothesis: After 3 failures, the pattern shows lower TFs (15m/30m/1h) have too much
noise and fee drag. 4h timeframe should naturally produce fewer, higher-quality signals
(20-50 trades/year target). This strategy combines:

1. 1D HMA trend bias: Very stable HTF direction filter. Only long if price>1d_HMA,
   only short if price<1d_HMA. Much more stable than 4h HMA for major trend.

2. 4h KAMA (Kaufman Adaptive): Adapts to volatility - fast in trends, slow in chop.
   Crossover of KAMA(10) and KAMA(30) signals trend changes with less whipsaw than EMA.

3. ADX(14) filter: Only trade when ADX>20 (trend present). Avoids range whipsaw.
   ADX>25 = strong trend (full size), ADX 20-25 = weak trend (half size).

4. Volume confirmation: Entry volume > 0.7*20bar_avg to filter fakeouts.

5. ATR-based position sizing and stoploss: Reduce size when vol is high,
   stoploss at 2.0*ATR trailing.

Why this should beat failed strategies:
- 4h TF = naturally fewer trades (less fee drag than 15m/30m/1h)
- 1d HMA = more stable trend bias than 4h HMA
- KAMA = adaptive to volatility (better than fixed EMA in 2022 crash)
- ADX filter = only trade when trend exists (avoids range chop)
- Discrete signals = less churn on every tiny change

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete, ATR-scaled
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_1d_hma_vol_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility - fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    n = len(close_s)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s - close_s.shift(er_period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    
    # Avoid division by zero
    er = change / volatility.replace(0, np.inf)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close_s.iloc[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s - high_s.shift(1)
    minus_dm = low_s.shift(1) - low_s
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.inf))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.inf))
    
    # DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    
    # ADX
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_dmi(high, low, close, period=14):
    """Calculate +DI and -DI for direction."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s - high_s.shift(1)
    minus_dm = low_s.shift(1) - low_s
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.inf))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.inf))
    
    return plus_di.values, minus_di.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    plus_di, minus_di = calculate_dmi(high, low, close, 14)
    
    # KAMA adaptive moving averages
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_slow = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
            continue
        
        # === 1D HMA TREND BIAS (very stable) ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === ADX TREND STRENGTH ===
        adx_value = adx_14[i]
        is_trending = adx_value > 20
        is_strong_trend = adx_value > 25
        
        # === DMI DIRECTION ===
        di_diff = plus_di[i] - minus_di[i]
        bullish_di = di_diff > 0
        bearish_di = di_diff < 0
        
        # === KAMA CROSSOVER ===
        # Fast KAMA crossing above slow KAMA
        kama_long = (kama_fast[i] > kama_slow[i]) and (kama_fast[i-1] <= kama_slow[i-1])
        # Fast KAMA crossing below slow KAMA
        kama_short = (kama_fast[i] < kama_slow[i]) and (kama_fast[i-1] >= kama_slow[i-1])
        
        # KAMA trend state (not just crossover)
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma[i]
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when volatility is high (protect in crashes)
        if i > 100:
            atr_median = np.nanmedian(atr_14[100:i])
            if atr_median > 0:
                atr_ratio = atr_14[i] / atr_median
            else:
                atr_ratio = 1.0
        else:
            atr_ratio = 1.0
        
        atr_ratio = np.clip(atr_ratio, 0.5, 2.0)  # Limit scaling
        size_multiplier = 1.0 / atr_ratio  # High vol = smaller size
        
        # Adjust size based on trend strength
        if is_strong_trend:
            trend_size_mult = 1.0
        elif is_trending:
            trend_size_mult = 0.7
        else:
            trend_size_mult = 0.5  # Weak trend = smaller size
        
        current_size = BASE_SIZE * size_multiplier * trend_size_mult
        current_size = np.clip(current_size, 0.15, 0.35)  # Keep in reasonable range
        
        # Round to discrete levels
        if current_size > 0.25:
            current_size = 0.30
        elif current_size > 0.15:
            current_size = 0.20
        else:
            current_size = 0.15
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: KAMA bullish + 1d HMA bull bias + ADX trending + volume
        if kama_bullish and bull_bias and is_trending and volume_confirmed:
            # Extra confirmation: +DI > -DI
            if bullish_di:
                new_signal = current_size
        
        # SHORT ENTRY: KAMA bearish + 1d HMA bear bias + ADX trending + volume
        if kama_bearish and bear_bias and is_trending and volume_confirmed:
            # Extra confirmation: -DI > +DI
            if bearish_di:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d bias turns bearish
            if position_side > 0 and bear_bias and adx_value > 20:
                trend_reversal = True
            # Exit short if 1d bias turns bullish
            if position_side < 0 and bull_bias and adx_value > 20:
                trend_reversal = True
            
            # Exit if ADX drops below 18 (trend ending)
            if adx_value < 18:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals