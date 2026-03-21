#!/usr/bin/env python3
"""
Experiment #032: 30m Supertrend + 4h HMA Regime + 1d Trend Filter + RSI Pullback
Hypothesis: 30m timeframe balances noise reduction with trade frequency.
4h HMA provides intermediate trend regime (proven in successful strategies).
1d HMA adds longer-term bias filter to avoid counter-trend trades in bear markets.
30m Supertrend gives precise entry timing with ATR-based direction.
RSI(14) pullback entries (RSI 40-60 range) ensure we buy dips in uptrends.
Z-score(20) filter avoids entering at price extremes (>2.0 std).
Multiple entry triggers ensure ≥10 trades while ATR stoploss (2.5x) protects capital.
Position sizing 0.30 with discrete levels minimizes fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_1d_hma_rsi_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for long-term bias
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    zscore = calculate_zscore(close, 20)
    
    # 30m HMA for short-term trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volume SMA
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
        # 4h trend filter (intermediate regime)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i]) and hma_4h_aligned[i] > 0
        four_hour_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        four_hour_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1d trend filter (long-term bias)
        hma_1d_valid = not np.isnan(hma_1d_aligned[i]) and hma_1d_aligned[i] > 0
        daily_bullish = hma_1d_valid and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_valid and close[i] < hma_1d_aligned[i]
        
        # 30m Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals (strongest entry)
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # HMA trend confirmation on 30m
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI pullback zones (not extreme)
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 60
        rsi_pullback_short = rsi[i] > 40 and rsi[i] < 60
        rsi_bullish_momentum = rsi[i] > 50
        rsi_bearish_momentum = rsi[i] < 50
        
        # Z-score filter (avoid extremes)
        zscore_valid = not np.isnan(zscore[i])
        zscore_neutral = zscore_valid and abs(zscore[i]) < 2.0
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Price position vs HMA21
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS
        # Trigger 1: Supertrend flip long + 4h bullish bias
        if st_flip_long and (four_hour_bullish or daily_bullish):
            new_signal = SIZE
        # Trigger 2: Supertrend long + HMA trend + RSI pullback + zscore ok
        elif st_long and hma_trend_long and rsi_pullback_long and zscore_neutral and price_above_hma:
            new_signal = SIZE
        # Trigger 3: 4h + 1d both bullish + Supertrend long (strong trend)
        elif four_hour_bullish and daily_bullish and st_long and vol_confirm:
            new_signal = SIZE
        # Trigger 4: RSI momentum + Supertrend + price above HMA
        elif rsi_bullish_momentum and st_long and price_above_hma and zscore_neutral:
            new_signal = SIZE
        # Trigger 5: Supertrend flip alone (catch strong moves)
        elif st_flip_long and vol_confirm:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: Supertrend flip short + 4h bearish bias
        if st_flip_short and (four_hour_bearish or daily_bearish):
            new_signal = -SIZE
        # Trigger 2: Supertrend short + HMA trend + RSI pullback + zscore ok
        elif st_short and hma_trend_short and rsi_pullback_short and zscore_neutral and price_below_hma:
            new_signal = -SIZE
        # Trigger 3: 4h + 1d both bearish + Supertrend short (strong trend)
        elif four_hour_bearish and daily_bearish and st_short and vol_confirm:
            new_signal = -SIZE
        # Trigger 4: RSI momentum + Supertrend + price below HMA
        elif rsi_bearish_momentum and st_short and price_below_hma and zscore_neutral:
            new_signal = -SIZE
        # Trigger 5: Supertrend flip alone (catch strong moves)
        elif st_flip_short and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > 0:
                    new_signal = 0.0
                elif close[i] > entry_price + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop > 0:
                    new_signal = 0.0
                elif close[i] < entry_price - 3.0 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
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