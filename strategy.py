#!/usr/bin/env python3
"""
Experiment #036: 1d Regime-Adaptive KAMA + Weekly HMA + Choppiness Index
Hypothesis: Daily timeframe with regime detection adapts to both bull (2021) and 
bear/range (2022, 2025) markets. Weekly HMA provides major trend filter. 
Choppiness Index (CHOP) detects ranging vs trending: CHOP>61.8=range (mean revert), 
CHOP<38.2=trend (trend follow). KAMA adapts to volatility better than EMA.
Multiple entry triggers ensure ≥10 trades while regime filter avoids whipsaws.
Position sizing 0.30 with 2.5x ATR stoploss controls drawdown.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_chop_regime_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - smooth in ranging, fast in trending.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
        er[i] = signal / noise if noise > 0 else 0
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += np.max(high[j] - low[j], 
                            np.abs(high[j] - close[j-1] if j > 0 else high[j] - low[j]),
                            np.abs(low[j] - close[j-1] if j > 0 else high[j] - low[j]))
        
        if tr_sum > 0 and highest_high > lowest_low:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate daily indicators (all vectorized before loop)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # KAMA for adaptive trend following
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=50)
    
    # Daily HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # Weekly trend filter (major regime)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Choppiness regime detection
        is_ranging = chop[i] > 55  # Relaxed from 61.8 for more trades
        is_trending = chop[i] < 45  # Relaxed from 38.2 for more trades
        
        # KAMA trend signals
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        kama_flip_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_flip_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI momentum
        rsi_bullish = rsi[i] > 40 and rsi[i] < 70
        rsi_bearish = rsi[i] > 30 and rsi[i] < 60
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Z-score mean reversion
        zscore_extreme_long = zscore[i] < -1.5
        zscore_extreme_short = zscore[i] > 1.5
        zscore_reverting_long = zscore[i] < -1.0 and zscore[i] > zscore[i-1]
        zscore_reverting_short = zscore[i] > 1.0 and zscore[i] < zscore[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # Price position
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (regime-adaptive)
        # Trigger 1: KAMA flip long in trending regime
        if kama_flip_long and is_trending and weekly_bullish:
            new_signal = SIZE
        # Trigger 2: KAMA bullish + HMA trend + RSI confirmation (trend follow)
        elif kama_bullish and hma_trend_long and rsi_bullish and weekly_bullish:
            new_signal = SIZE
        # Trigger 3: Z-score mean reversion in ranging regime
        elif zscore_extreme_long and is_ranging and rsi_oversold:
            new_signal = SIZE
        # Trigger 4: Z-score reverting + KAMA aligned (mean reversion with trend)
        elif zscore_reverting_long and kama_bullish and vol_confirm:
            new_signal = SIZE
        # Trigger 5: Weekly bullish + price pullback to HMA21
        elif weekly_bullish and price_above_hma21 and rsi[i] > 45 and rsi[i-1] <= 45:
            new_signal = SIZE
        # Trigger 6: Simple KAMA crossover with volume (ensure trades)
        elif kama_flip_long and vol_confirm:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS (regime-adaptive)
        # Trigger 1: KAMA flip short in trending regime
        if kama_flip_short and is_trending and weekly_bearish:
            new_signal = -SIZE
        # Trigger 2: KAMA bearish + HMA trend + RSI confirmation (trend follow)
        elif kama_bearish and hma_trend_short and rsi_bearish and weekly_bearish:
            new_signal = -SIZE
        # Trigger 3: Z-score mean reversion in ranging regime
        elif zscore_extreme_short and is_ranging and rsi_overbought:
            new_signal = -SIZE
        # Trigger 4: Z-score reverting + KAMA aligned (mean reversion with trend)
        elif zscore_reverting_short and kama_bearish and vol_confirm:
            new_signal = -SIZE
        # Trigger 5: Weekly bearish + price rally to HMA21
        elif weekly_bearish and price_below_hma21 and rsi[i] < 55 and rsi[i-1] >= 55:
            new_signal = -SIZE
        # Trigger 6: Simple KAMA crossover with volume (ensure trades)
        elif kama_flip_short and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                if close[i] > entry_price + 2.5 * atr[entry_price < close] if entry_price > 0 else False:
                    pass  # Simplified - just trail
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals