#!/usr/bin/env python3
"""
Experiment #052: 4h Adaptive Regime Strategy with 1d/1w Trend Filter
Hypothesis: 4h timeframe captures medium-term swings while 1d/1w HTF provides regime bias.
Key insight: Use 1w HMA for primary trend direction, 1d ADX for regime (trend vs range).
In trending regimes: follow 4h HMA direction with RSI pullback entries.
In ranging regimes: mean revert at Bollinger extremes with CRSI confirmation.
Position sizing: 0.25 base, 0.35 for strong trend alignment, stoploss at 2.5*ATR.
Why this might work: 4h has enough trades (50-100/year) while avoiding 15m noise.
HTF filter prevents counter-trend trades that destroy Sharpe in bear markets.
Must generate 10+ trades on train - entry conditions loosened vs failed experiments.
Timeframe: 4h (REQUIRED for exp#052), HTF: 1d and 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adaptive_regime_1d_1w_hma_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / (sma + 1e-10)
    return upper, lower, bb_width, sma

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent Rank - where current close ranks in last 100 days
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3
    
    return crsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    ema_1d_50 = calculate_ema(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_3 = calculate_rsi(close, 3)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    bb_upper, bb_lower, bb_width, bb_sma = calculate_bollinger(close, 20, 2.0)
    zscore = calculate_zscore(close, 20)
    crsi = calculate_crsi(close, 3, 2, 100)
    kama = calculate_kama(close, 10, 2, 30)
    hma_4h = calculate_hma(close, 21)
    
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
    
    for i in range(250, n):
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
        
        # === HTF TREED BIAS (1w primary, 1d secondary) ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # Strong trend when 1w and 1d agree
        strong_bull = bull_trend_1w and bull_trend_1d
        strong_bear = bear_trend_1w and bear_trend_1d
        
        # === 4h TREED CONFIRMATION ===
        bull_trend_4h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_4h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # === ADX REGIME DETECTION ===
        trending_regime = adx[i] > 22
        ranging_regime = adx[i] < 18
        
        # === DI MOMENTUM ===
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # === CRSI MEAN REVERSION ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 20
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 80
        
        # === BOLLINGER BAND SIGNALS ===
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.005
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.995
        bb_squeeze = bb_width[i] < np.nanpercentile(bb_width[max(0,i-100):i], 25) if i > 100 else False
        
        # === Z-SCORE EXTREMES ===
        zscore_extreme_long = zscore[i] < -1.5
        zscore_extreme_short = zscore[i] > 1.5
        
        # === KAMA TREED ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === HMA CROSSOVER (4h) ===
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_4h[i]) and not np.isnan(hma_4h[i-1]):
            hma_cross_long = hma_4h[i] > ema_50[i] and hma_4h[i-1] <= ema_50[i-1]
            hma_cross_short = hma_4h[i] < ema_50[i] and hma_4h[i-1] >= ema_50[i-1]
        
        # === PULLBACK TO EMA21 ===
        pullback_long = close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.98
        pullback_short = close[i] >= ema_21[i] * 0.98 and close[i] <= ema_21[i] * 1.02
        
        new_signal = 0.0
        
        # === TRENDING REGIME (ADX > 22) ===
        if trending_regime:
            # Long entries in bullish trend
            if strong_bull or (bull_trend_1d and bull_trend_4h):
                if pullback_long and rsi_oversold and di_bullish:
                    new_signal = SIZE_STRONG
                elif hma_cross_long and kama_bullish:
                    new_signal = SIZE_BASE
                elif rsi_oversold and close[i] > bb_sma[i]:
                    new_signal = SIZE_HALF
            
            # Short entries in bearish trend
            elif strong_bear or (bear_trend_1d and bear_trend_4h):
                if pullback_short and rsi_overbought and di_bearish:
                    new_signal = -SIZE_STRONG
                elif hma_cross_short and kama_bearish:
                    new_signal = -SIZE_BASE
                elif rsi_overbought and close[i] < bb_sma[i]:
                    new_signal = -SIZE_HALF
        
        # === RANGING REGIME (ADX < 18) ===
        elif ranging_regime:
            # Mean reversion long at support
            if crsi_oversold and price_at_lower_bb:
                if bull_trend_1w or zscore_extreme_long:
                    new_signal = SIZE_BASE
            elif rsi_oversold and zscore_extreme_long and price_at_lower_bb:
                new_signal = SIZE_HALF
            
            # Mean reversion short at resistance
            if crsi_overbought and price_at_upper_bb:
                if bear_trend_1w or zscore_extreme_short:
                    new_signal = -SIZE_BASE
            elif rsi_overbought and zscore_extreme_short and price_at_upper_bb:
                new_signal = -SIZE_HALF
        
        # === TRANSITION REGIME (18 <= ADX <= 22) ===
        else:
            # Conservative entries with HTF confirmation only
            if strong_bull and hma_cross_long:
                new_signal = SIZE_HALF
            elif strong_bear and hma_cross_short:
                new_signal = -SIZE_HALF
            elif crsi_oversold and price_at_lower_bb and bull_trend_1w:
                new_signal = SIZE_HALF
            elif crsi_overbought and price_at_upper_bb and bear_trend_1w:
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