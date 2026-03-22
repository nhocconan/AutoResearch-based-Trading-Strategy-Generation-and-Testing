#!/usr/bin/env python3
"""
Experiment #330: 1d Trend-Following with 1w Meta-Trend and Regime Filter

Hypothesis: After 283 failed strategies, the pattern is clear:
1. Mean reversion fails catastrophically on crypto (Sharpe -3 to -15)
2. Simple trend-following with HTF bias works best
3. 1d timeframe needs wider stops and fewer but higher-quality trades
4. Regime filter (Choppiness Index) prevents whipsaw in ranging markets

This strategy combines:
1. 1w HMA(21) for meta-trend direction (only trade WITH meta-trend)
2. 1d EMA(8)/EMA(21) crossover for entry timing
3. Choppiness Index(14) regime filter: CHOP<45 = trending, CHOP>55 = range
4. ADX(14)>18 for trend strength confirmation
5. ATR(14)*3 trailing stoploss (wider for daily volatility)
6. Position sizing: 0.25 base, 0.35 with strong trend + regime confirm

Why this should work on 1d:
- Daily bars filter out intraday noise
- 1w meta-trend prevents counter-trend trades in strong moves
- CHOP filter avoids entering during choppy consolidation
- Wider ATR stop (3x vs 2.5x) reduces premature exits on daily volatility
- Discrete sizing minimizes fee churn on signal changes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trend_1w_hma_chop_regime_atr_v1"
timeframe = "1d"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_val = highest_high - lowest_low
        
        if range_val > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_val) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    adx = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME META-TREND ===
        # 1w HMA = meta-trend direction (REQUIRED - only trade WITH meta-trend)
        bull_meta_trend = close[i] > hma_1w_aligned[i]
        bear_meta_trend = close[i] < hma_1w_aligned[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP < 45 = trending market (allow entries)
        # CHOP > 55 = ranging market (reduce size or skip)
        # CHOP 45-55 = neutral
        trending_regime = chop[i] < 45
        ranging_regime = chop[i] > 55
        
        # === TREND STRENGTH ===
        # ADX > 18 = trending (moderate threshold for trade generation)
        trending = adx[i] > 18
        strong_trend = adx[i] > 25
        
        # === EMA CROSSOVER ===
        # Fast EMA crosses above slow EMA = bullish
        ema_bullish = ema_fast[i] > ema_slow[i]
        # Fast EMA crosses below slow EMA = bearish
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # Check for actual crossover (not just state)
        ema_cross_bull = ema_bullish and (i > 0 and ema_fast[i-1] <= ema_slow[i-1])
        ema_cross_bear = ema_bearish and (i > 0 and ema_fast[i-1] >= ema_slow[i-1])
        
        # Also allow entry if already in trend state (not just crossover)
        ema_trend_bull = ema_bullish
        ema_trend_bear = ema_bearish
        
        # === DETERMINE POSITION SIZE ===
        # Base size with strong confirmation = SIZE_STRONG
        # Base size without strong confirm = SIZE_BASE
        # Ranging regime = reduce to SIZE_BASE or skip
        if ranging_regime:
            position_size = SIZE_BASE * 0.5  # reduce in choppy markets
        elif strong_trend and trending_regime:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: 1w meta-trend up + EMA bullish + ADX trending + not ranging
        # Relaxed: allow entry in neutral regime too
        long_conditions = (
            bull_meta_trend and
            ema_trend_bull and
            trending and
            not ranging_regime
        )
        
        # SHORT: 1w meta-trend down + EMA bearish + ADX trending + not ranging
        short_conditions = (
            bear_meta_trend and
            ema_trend_bear and
            trending and
            not ranging_regime
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 3.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === META-TREND REVERSAL EXIT ===
        # Exit if 1w HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_meta_trend:
                new_signal = 0.0
            if position_side < 0 and bull_meta_trend:
                new_signal = 0.0
        
        # === EMA REVERSAL EXIT ===
        # Exit if EMA crossover flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and ema_bearish:
                new_signal = 0.0
            if position_side < 0 and ema_bullish:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals