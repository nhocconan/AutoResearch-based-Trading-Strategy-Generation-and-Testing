#!/usr/bin/env python3
"""
Experiment #008: 30m Adaptive Regime Strategy with 4h Trend + Daily Filter
Hypothesis: 30m timeframe captures swing trades with less noise than 15m.
Regime-adaptive logic: trending markets use trend-following, ranging markets use mean-reversion.
4h HMA provides intermediate trend filter, Daily SMA200 provides major regime filter.
Bollinger Band Width percentile detects trending (low BW) vs ranging (high BW) regimes.
ATR-based stoploss (2.5x) protects against crashes. Position sizing capped at 0.30.
Relaxed entry conditions to ensure ≥10 trades/symbol on train data across all market regimes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adaptive_4h_v1"
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

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bw = (upper - lower) / sma
    bw = np.nan_to_num(bw, nan=0.0)
    return upper, lower, sma, bw

def calculate_bw_percentile(bw, lookback=100):
    """Calculate Bollinger Band Width percentile for regime detection."""
    bw_percentile = np.zeros(len(bw))
    for i in range(lookback, len(bw)):
        window = bw[i-lookback:i+1]
        bw_percentile[i] = np.sum(window <= bw[i]) / len(window) * 100
    return bw_percentile

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    supertrend[0] = upper[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            supertrend[i] = upper[i]
            direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d SMA200 for major trend
    sma_1d = calculate_sma(df_1d['close'].values, 200)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_sma, bb_bw = calculate_bollinger_bands(close, 20, 2.0)
    bw_percentile = calculate_bw_percentile(bb_bw, 100)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(150, n):
        # Regime detection: low BW percentile = trending, high = ranging
        trending_regime = bw_percentile[i] < 40  # Bottom 40% = trending
        ranging_regime = bw_percentile[i] > 60   # Top 40% = ranging
        
        # 4h trend filter
        hma_4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        hma_4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        
        # 1d major trend filter (relaxed for more trades)
        daily_bullish = sma_1d_aligned[i] > 0 and close[i] > sma_1d_aligned[i] * 0.95
        daily_bearish = sma_1d_aligned[i] > 0 and close[i] < sma_1d_aligned[i] * 1.05
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # MACD momentum
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = rsi[i] > 35 and rsi[i] < 65
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Entry logic - regime adaptive
        new_signal = 0.0
        
        # TRENDING REGIME: Follow 4h trend with momentum
        if trending_regime:
            # Long: 4h bullish + Supertrend long + MACD positive
            if hma_4h_bullish and st_long and macd_bullish and vol_confirm:
                new_signal = SIZE
            # Short: 4h bearish + Supertrend short + MACD negative
            elif hma_4h_bearish and st_short and macd_bearish and vol_confirm:
                new_signal = -SIZE
            # Supertrend flip entries
            elif st_direction[i] == 1 and st_direction[i-1] == -1 and hma_4h_bullish:
                new_signal = SIZE
            elif st_direction[i] == -1 and st_direction[i-1] == 1 and hma_4h_bearish:
                new_signal = -SIZE
        
        # RANGING REGIME: Mean reversion with RSI extremes
        elif ranging_regime:
            # Long: RSI oversold + price near BB lower + 4h not strongly bearish
            if rsi_oversold and close[i] < bb_lower[i] * 1.005 and not hma_4h_bearish:
                new_signal = SIZE
            # Short: RSI overbought + price near BB upper + 4h not strongly bullish
            elif rsi_overbought and close[i] > bb_upper[i] * 0.995 and not hma_4h_bullish:
                new_signal = -SIZE
            # RSI reversal from extremes
            elif rsi[i] < 35 and rsi[i-1] < rsi[i] and close[i] > bb_sma[i] * 0.98:
                new_signal = SIZE
            elif rsi[i] > 65 and rsi[i-1] > rsi[i] and close[i] < bb_sma[i] * 1.02:
                new_signal = -SIZE
        
        # NEUTRAL REGIME: Use daily trend filter
        else:
            if daily_bullish and st_long and rsi_neutral:
                new_signal = SIZE
            elif daily_bearish and st_short and rsi_neutral:
                new_signal = -SIZE
            elif st_direction[i] == 1 and st_direction[i-1] == -1 and vol_confirm:
                new_signal = SIZE
            elif st_direction[i] == -1 and st_direction[i-1] == 1 and vol_confirm:
                new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, close[i] - 2.5 * atr[i])
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                elif close[i] > entry_price[i-1] + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                trailing_stop[i] = min(trailing_stop[i-1] if i > 0 else 999999, close[i] + 2.5 * atr[i])
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                elif close[i] < entry_price[i-1] - 3.0 * atr[i] and signals[i-1] == -SIZE:
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
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0
        
        signals[i] = new_signal
    
    return signals