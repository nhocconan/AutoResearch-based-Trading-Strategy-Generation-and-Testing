#!/usr/bin/env python3
"""
Experiment #058: 4h Regime-Adaptive Strategy with 1d/1w HMA Filter
Hypothesis: 4h timeframe captures medium-term trends while avoiding noise.
Key insight: Use Choppiness Index to switch between trend-following (CHOP<38.2) 
and mean-reversion (CHOP>61.8) regimes. 1d/1w HMA provides macro trend bias.
Entry on RSI pullbacks in trend regime, BB extremes in range regime.
Why this might work: Adapts to market conditions instead of one-size-fits-all.
4h has shown promise in experiment history when combined with HTF filters.
Must generate 10+ trades - entry conditions loosened vs failed experiments.
Timeframe: 4h (REQUIRED for exp#058), HTF: 1d and 1w via mtf_data helper.
Position sizing: 0.20-0.35 discrete, stoploss at 2.5*ATR, leverage=1.0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_1d_1w_hma_chop_v2"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(calculate_atr(high[i-period+1:i+1], low[i-period+1:i+1], close[i-period+1:i+1], period))
        
        if highest_high - lowest_low > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(period)
    
    return chop

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)
    
    supertrend[period] = upper_band[period]
    direction[period] = 1
    
    for i in range(period + 1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    return supertrend, direction

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    kama = calculate_kama(close, 10, 2, 30)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Choppiness Index for regime detection
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # Z-score for extreme moves
    zscore = calculate_zscore(close, 20)
    
    # HMA on 4h for faster trend
    hma_4h = calculate_hma(close, 21)
    hma_4h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        transition_regime = 38.2 <= chop[i] <= 61.8
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        bull_trend_4h = hma_4h_fast[i] > hma_4h[i]
        bear_trend_4h = hma_4h_fast[i] < hma_4h[i]
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = ema_21[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === TREND STRENGTH ===
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # === PRICE POSITION ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Price near EMA21 (pullback entry zone)
        price_near_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        price_near_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        # Bollinger Band position
        price_at_bb_lower = close[i] <= bb_lower[i] * 1.01
        price_at_bb_upper = close[i] >= bb_upper[i] * 0.99
        price_near_bb_mid = abs(close[i] - bb_mid[i]) < (bb_upper[i] - bb_lower[i]) * 0.15
        
        # Z-score extremes
        zscore_extreme_low = zscore[i] < -1.5
        zscore_extreme_high = zscore[i] > 1.5
        
        # === KAMA ADAPTIVE TREND ===
        kama_bullish = close[i] > kama[i] if not np.isnan(kama[i]) else False
        kama_bearish = close[i] < kama[i] if not np.isnan(kama[i]) else False
        
        # === RSI CONDITIONS (looser for more trades) ===
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_neutral_long = 40 <= rsi[i] <= 60
        rsi_neutral_short = 40 <= rsi[i] <= 60
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 70
        rsi_momentum_short = rsi[i] < 55 and rsi[i] > 30
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_4h_fast[i]) and not np.isnan(hma_4h_fast[i-1]):
            hma_cross_long = hma_4h_fast[i] > hma_4h[i] and hma_4h_fast[i-1] <= hma_4h[i-1]
            hma_cross_short = hma_4h_fast[i] < hma_4h[i] and hma_4h_fast[i-1] >= hma_4h[i-1]
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) - Trend Following ===
        if trending_regime:
            # LONG: All HTF bullish + pullback entry
            if bull_trend_1w and bull_trend_1d:
                if rsi_neutral_long and price_near_ema21_long:
                    if di_bullish or st_bullish or bull_trend_4h:
                        new_signal = SIZE_STRONG if strong_trend else SIZE_BASE
                elif hma_cross_long and rsi_momentum_long:
                    new_signal = SIZE_BASE
                elif kama_bullish and above_sma200 and rsi[i] > 40:
                    new_signal = SIZE_HALF
            
            # SHORT: All HTF bearish + pullback entry
            if bear_trend_1w and bear_trend_1d:
                if rsi_neutral_short and price_near_ema21_short:
                    if di_bearish or st_bearish or bear_trend_4h:
                        new_signal = -SIZE_STRONG if strong_trend else -SIZE_BASE
                elif hma_cross_short and rsi_momentum_short:
                    new_signal = -SIZE_BASE
                elif kama_bearish and below_sma200 and rsi[i] < 60:
                    new_signal = -SIZE_HALF
        
        # === RANGING REGIME (CHOP > 61.8) - Mean Reversion ===
        if ranging_regime:
            # LONG: BB lower + RSI oversold + HTF not strongly bearish
            if price_at_bb_lower and rsi_oversold:
                if not bear_trend_1w or zscore_extreme_low:
                    new_signal = SIZE_HALF
            # SHORT: BB upper + RSI overbought + HTF not strongly bullish
            if price_at_bb_upper and rsi_overbought:
                if not bull_trend_1w or zscore_extreme_high:
                    new_signal = -SIZE_HALF
            
            # Z-score extreme mean reversion
            if zscore_extreme_low and rsi[i] < 40:
                if bull_trend_1d or not bear_trend_1w:
                    new_signal = SIZE_BASE
            if zscore_extreme_high and rsi[i] > 60:
                if bear_trend_1d or not bull_trend_1w:
                    new_signal = -SIZE_BASE
        
        # === TRANSITION REGIME (38.2 <= CHOP <= 61.8) - Reduced Size ===
        if transition_regime:
            # Only take strong signals with HTF confirmation
            if bull_trend_1w and bull_trend_1d and hma_cross_long:
                if rsi_momentum_long:
                    new_signal = SIZE_HALF
            if bear_trend_1w and bear_trend_1d and hma_cross_short:
                if rsi_momentum_short:
                    new_signal = -SIZE_HALF
        
        # === SUPERTREND FLIP SIGNALS (additional entry paths) ===
        if st_bullish and bull_trend_1d and rsi_momentum_long:
            if new_signal == 0.0:
                new_signal = SIZE_HALF
        if st_bearish and bear_trend_1d and rsi_momentum_short:
            if new_signal == 0.0:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals