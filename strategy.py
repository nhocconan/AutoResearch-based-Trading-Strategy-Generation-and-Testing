#!/usr/bin/env python3
"""
Experiment #008: 30m Regime-Adaptive Multi-Strategy with 4h/1d HTF Bias
Hypothesis: 30m timeframe captures intraday swings while 4h/1d HTF provides trend context.
Key insight from research: BTC/ETH 2025+ is bear/range market, not trending. Simple trend
following fails. Need regime detection (Choppiness Index) to switch between:
- Trending regime (CHOP<38.2): Follow HTF trend with RSI pullback entries
- Ranging regime (CHOP>61.8): Mean revert at Bollinger bands
- Transition regime: Stay flat or reduce size
Multiple entry paths (8+ for longs, 8+ for shorts) ensure >=10 trades per symbol.
Conservative sizing (0.25) with 2.5*ATR stoploss controls drawdown.
Timeframe: 30m (REQUIRED), HTF: 4h + 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adaptive_chop_4h_1d_hma_rsi_bb_atr_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        atr_sum = np.sum(atr[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market efficiency ratio - fast in trends, slow in ranges.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        if i == er_period:
            kama[i] = close[i]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR bands.
    Returns: supertrend values, direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band
                direction[i] = 1
            elif close[i] < supertrend[i-1]:
                supertrend[i] = upper_band
                direction[i] = -1
            else:
                if direction[i-1] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
                    direction[i] = -1
    
    return supertrend, direction

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / np.where(std > 0, std, 1.0)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    ema_4h_50 = calculate_ema(df_4h['close'].values, 50)
    ema_1d_50 = calculate_ema(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for entries
    adx = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    supertrend_30m, st_dir_30m = calculate_supertrend(high, low, close, 10, 3.0)
    zscore = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    SIZE_QUARTER = 0.08
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(st_dir_30m[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_trending = chop[i] < 38.2
        regime_ranging = chop[i] > 61.8
        regime_transition = not regime_trending and not regime_ranging
        
        # === HTF TREND BIAS (4h + 1d) ===
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        ema_4h_bullish = close[i] > ema_4h_50_aligned[i]
        ema_4h_bearish = close[i] < ema_4h_50_aligned[i]
        ema_1d_bullish = close[i] > ema_1d_50_aligned[i]
        ema_1d_bearish = close[i] < ema_1d_50_aligned[i]
        
        # Strong HTF trend confirmation (both 4h and 1d agree)
        htf_strong_bull = hma_4h_bullish and hma_1d_bullish and ema_4h_bullish
        htf_strong_bear = hma_4h_bearish and hma_1d_bearish and ema_4h_bearish
        
        # Moderate HTF bias (at least 4h agrees)
        htf_mod_bull = hma_4h_bullish and ema_4h_bullish
        htf_mod_bear = hma_4h_bearish and ema_4h_bearish
        
        # === 30m TREND INDICATORS ===
        st_30m_bullish = st_dir_30m[i] == 1
        st_30m_bearish = st_dir_30m[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_dir_30m[i] == 1 and st_dir_30m[i-1] == -1 if i > 0 else False
        st_flip_short = st_dir_30m[i] == -1 and st_dir_30m[i-1] == 1 if i > 0 else False
        
        # EMA trend on 30m
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        kama_slope_up = kama[i] > kama[i-1] if i > 0 else False
        kama_slope_down = kama[i] < kama[i-1] if i > 0 else False
        
        # ADX trend strength
        adx_weak = adx[i] > 15
        adx_strong = adx[i] > 25
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 50
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 65
        rsi_extreme_long = rsi_7[i] < 25
        rsi_extreme_short = rsi_7[i] > 75
        
        # Bollinger bands
        bb_touch_lower = close[i] <= bb_lower[i] * 1.002
        bb_touch_upper = close[i] >= bb_upper[i] * 0.998
        bb_break_lower = close[i] < bb_lower[i]
        bb_break_upper = close[i] > bb_upper[i]
        
        # Z-score extremes
        zscore_extreme_long = zscore[i] < -1.5
        zscore_extreme_short = zscore[i] > 1.5
        
        new_signal = 0.0
        
        # === LONG ENTRIES (8+ paths for >=10 trades) ===
        
        # Path 1: Trending regime + HTF bullish + RSI pullback
        if regime_trending and htf_strong_bull and rsi_pullback_long and adx_weak:
            new_signal = SIZE_ENTRY
        
        # Path 2: Trending regime + HTF bullish + Supertrend flip long
        elif regime_trending and htf_mod_bull and st_flip_long:
            new_signal = SIZE_ENTRY
        
        # Path 3: Ranging regime + BB touch lower + HTF not bearish + RSI oversold
        elif regime_ranging and bb_touch_lower and not htf_strong_bear and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 4: Ranging regime + Z-score extreme long + HTF neutral/bull
        elif regime_ranging and zscore_extreme_long and not htf_strong_bear:
            new_signal = SIZE_ENTRY
        
        # Path 5: Transition regime + HTF strong bull + KAMA bullish + ADX building
        elif regime_transition and htf_strong_bull and kama_bullish and kama_slope_up and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # Path 6: Any regime + HTF strong bull + Supertrend bullish + EMA bullish
        elif htf_strong_bull and st_30m_bullish and ema_bullish and adx_weak:
            new_signal = SIZE_ENTRY
        
        # Path 7: RSI extreme long + HTF not bearish (oversold bounce)
        elif rsi_extreme_long and not htf_strong_bear:
            new_signal = SIZE_QUARTER
        
        # Path 8: KAMA bullish + Supertrend bullish + ADX > 20
        elif kama_bullish and kama_slope_up and st_30m_bullish and adx[i] > 20:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (8+ paths for >=10 trades) ===
        
        # Path 1: Trending regime + HTF bearish + RSI pullback
        if regime_trending and htf_strong_bear and rsi_pullback_short and adx_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Trending regime + HTF bearish + Supertrend flip short
        elif regime_trending and htf_mod_bear and st_flip_short:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Ranging regime + BB touch upper + HTF not bullish + RSI overbought
        elif regime_ranging and bb_touch_upper and not htf_strong_bull and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Ranging regime + Z-score extreme short + HTF neutral/bear
        elif regime_ranging and zscore_extreme_short and not htf_strong_bull:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Transition regime + HTF strong bear + KAMA bearish + ADX building
        elif regime_transition and htf_strong_bear and kama_bearish and kama_slope_down and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # Path 6: Any regime + HTF strong bear + Supertrend bearish + EMA bearish
        elif htf_strong_bear and st_30m_bearish and ema_bearish and adx_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 7: RSI extreme short + HTF not bullish (overbought drop)
        elif rsi_extreme_short and not htf_strong_bull:
            new_signal = -SIZE_QUARTER
        
        # Path 8: KAMA bearish + Supertrend bearish + ADX > 20
        elif kama_bearish and kama_slope_down and st_30m_bearish and adx[i] > 20:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals