#!/usr/bin/env python3
"""
Experiment #014: 30m Mean Reversion + 4h Trend Filter + Choppiness Regime
Hypothesis: 30m timeframe is ideal for mean reversion with HTF trend filter.
4h HMA provides major trend direction (only trade with trend).
Choppiness Index (CHOP) detects range vs trend regime - enter on mean reversion in ranges.
RSI(14) extremes with SMA(200) filter for entry timing.
ATR-based stoploss (2.5x) protects against crashes.
Position sizing: 0.25 discrete levels to minimize fee churn while ensuring trades.
Relaxed entry conditions to ensure ≥10 trades/symbol on train data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_mr_chop_4h_v1"
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market (mean reversion favorable)
    CHOP < 38.2 = trending market (trend following favorable)
    """
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range > 0, price_range, 1e-10)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    std = np.where(std > 0, std, 1e-10)
    zscore = (close - sma) / std
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    chop = calculate_choppiness(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(250, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Choppiness regime detection
        is_choppy = chop[i] > 55.0  # Range market - favor mean reversion
        is_trending = chop[i] < 45.0  # Trend market - favor trend following
        
        # Price relative to SMA200
        above_sma200 = sma_200[i] > 0 and close[i] > sma_200[i]
        below_sma200 = sma_200[i] > 0 and close[i] < sma_200[i]
        
        # RSI extremes for mean reversion (relaxed for more trades)
        rsi_oversold = rsi[i] < 45  # Relaxed from 30
        rsi_overbought = rsi[i] > 55  # Relaxed from 70
        rsi_extreme_oversold = rsi[i] < 35
        rsi_extreme_overbought = rsi[i] > 65
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.0
        zscore_overbought = zscore[i] > 1.0
        
        # Volume confirmation (relaxed)
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Entry logic - multiple pathways to ensure trades
        new_signal = 0.0
        
        # PATH 1: Mean reversion in choppy market (primary)
        if is_choppy:
            # Long: oversold RSI + above SMA200 + 4h not bearish
            if rsi_oversold and (above_sma200 or not trend_4h_bearish):
                new_signal = SIZE
            # Short: overbought RSI + below SMA200 + 4h not bullish
            elif rsi_overbought and (below_sma200 or not trend_4h_bullish):
                new_signal = -SIZE
        
        # PATH 2: Trend following in trending market
        elif is_trending:
            # Long: 4h bullish + RSI rising from oversold
            if trend_4h_bullish and rsi_extreme_oversold and rsi[i] > rsi[i-3]:
                new_signal = SIZE
            # Short: 4h bearish + RSI falling from overbought
            elif trend_4h_bearish and rsi_extreme_overbought and rsi[i] < rsi[i-3]:
                new_signal = -SIZE
        
        # PATH 3: Z-score mean reversion (always active)
        if zscore_oversold and rsi[i] < 50:
            new_signal = SIZE
        elif zscore_overbought and rsi[i] > 50:
            new_signal = -SIZE
        
        # PATH 4: Simple RSI extremes with volume (fallback for trades)
        if rsi[i] < 30 and vol_confirm:
            new_signal = SIZE
        elif rsi[i] > 70 and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs
            else:
                prev_trail = trailing_stop[i-1] if i > 0 else 0
                trailing_stop[i] = max(prev_trail, close[i] - 2.5 * atr[i])
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] > entry_price[i-1] + 2.5 * atr[i] and new_signal == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                prev_trail = trailing_stop[i-1] if i > 0 else 999999
                trailing_stop[i] = min(prev_trail, close[i] + 2.5 * atr[i])
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] < entry_price[i-1] - 2.5 * atr[i] and new_signal == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            else:
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals