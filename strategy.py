#!/usr/bin/env python3
"""
Experiment #002: 30m Multi-Timeframe Trend + Mean Reversion Hybrid
Hypothesis: 30m timeframe balances noise reduction with trade frequency.
4h HMA provides major trend direction (bull/bear regime filter).
30m RSI pullback entries in trend direction + MACD confirmation.
Choppiness Index detects range vs trend regime for adaptive logic.
ATR-based stoploss (2.5x) protects against crashes like 2022.
Position sizing capped at 0.30 with discrete levels to minimize fee churn.
This should generate 30-60 trades/year with positive Sharpe on ALL symbols.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_chop_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    atr = calculate_atr(high, low, close, period)
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    tr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (close - sma) / std
    zscore = np.nan_to_num(zscore, nan=0.0)
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
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    chop = calculate_choppiness(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        is_trending = chop[i] < 55  # Below 55 = trending market
        is_ranging = chop[i] > 62   # Above 62 = ranging market
        
        # 30m EMA trend
        ema_trend_long = ema_fast[i] > ema_slow[i] and close[i] > ema_50[i]
        ema_trend_short = ema_fast[i] < ema_slow[i] and close[i] < ema_50[i]
        
        # RSI conditions - relaxed for more trades
        rsi_long = rsi[i] > 35 and rsi[i] < 70
        rsi_short = rsi[i] < 65 and rsi[i] > 30
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55 and rsi[i] > rsi[i-1] if i > 0 else False
        rsi_pullback_short = rsi[i] < 60 and rsi[i] > 45 and rsi[i] < rsi[i-1] if i > 0 else False
        
        # MACD confirmation
        macd_long = macd_hist[i] > 0 and (i < 1 or macd_hist[i] >= macd_hist[i-1])
        macd_short = macd_hist[i] < 0 and (i < 1 or macd_hist[i] <= macd_hist[i-1])
        
        # Volume confirmation (relaxed)
        vol_confirm = vol_sma[i] > 0 and volume[i] > vol_sma[i] * 0.7
        
        # Z-score mean reversion filter
        zscore_ok_long = zscore[i] > -2.5  # Not extremely oversold
        zscore_ok_short = zscore[i] < 2.5  # Not extremely overbought
        
        # Entry logic - multiple pathways to ensure trades
        new_signal = 0.0
        
        # LONG ENTRY PATHWAYS
        # Path 1: Trending market + 4h bullish + EMA trend + RSI ok
        if is_trending and trend_4h_bullish and ema_trend_long and rsi_long:
            new_signal = SIZE
        # Path 2: 4h bullish + RSI pullback + MACD confirmation
        elif trend_4h_bullish and rsi_pullback_long and macd_long:
            new_signal = SIZE
        # Path 3: Range market + mean reversion long (RSI oversold)
        elif is_ranging and rsi[i] < 40 and zscore[i] < -1.0:
            new_signal = SIZE
        # Path 4: EMA crossover with 4h support
        elif trend_4h_bullish and ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and rsi[i] > 40:
            new_signal = SIZE
        
        # SHORT ENTRY PATHWAYS
        # Path 1: Trending market + 4h bearish + EMA trend + RSI ok
        elif is_trending and trend_4h_bearish and ema_trend_short and rsi_short:
            new_signal = -SIZE
        # Path 2: 4h bearish + RSI pullback + MACD confirmation
        elif trend_4h_bearish and rsi_pullback_short and macd_short:
            new_signal = -SIZE
        # Path 3: Range market + mean reversion short (RSI overbought)
        elif is_ranging and rsi[i] > 60 and zscore[i] > 1.0:
            new_signal = -SIZE
        # Path 4: EMA crossover with 4h resistance
        elif trend_4h_bearish and ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1] and rsi[i] < 60:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take partial profit at 3R
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if signals[i-1] == SIZE:  # Only reduce if at full size
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take partial profit at 3R
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if signals[i-1] == -SIZE:  # Only reduce if at full size
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                entry_price[i] = entry_price[i-1]
                highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
                lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else close[i]
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals