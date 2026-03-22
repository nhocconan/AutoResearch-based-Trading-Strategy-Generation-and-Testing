#!/usr/bin/env python3
"""
Experiment #037: 15m Mean Reversion + 1h HMA Trend + Volume Confirmation
Hypothesis: 15m timeframe captures intraday mean reversion opportunities that 1h/4h miss.
Combined with 1h HMA trend filter to avoid counter-trend trades during strong moves.
Volume spike confirmation ensures entries have institutional participation.
Z-score filter identifies extreme deviations from recent mean (mean reversion edge).
ATR trailing stop at 2.5*ATR limits drawdown during volatile periods.
Timeframe: 15m (REQUIRED), HTF: 1h via mtf_data helper (faster response than 4h for 15m entries).
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: Z-score(20) < -2.0 + RSI(7) < 35 = strong oversold with statistical edge.
Looser entry conditions to ensure ≥10 trades/symbol on train data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_zscore_rsi_1h_hma_vol_v1"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    zscore = zscore.fillna(0.0).values
    return zscore

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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio relative to rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg
    vol_ratio = vol_ratio.fillna(1.0).values
    return vol_ratio

def calculate_ema(close, period):
    """Calculate exponential moving average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # 1h trend bias (HTF)
        bull_trend = close[i] > hma_1h_aligned[i]
        bear_trend = close[i] < hma_1h_aligned[i]
        
        # RSI signals (mean reversion)
        rsi_oversold = rsi_7[i] < 35
        rsi_overbought = rsi_7[i] > 65
        rsi_extreme_oversold = rsi_7[i] < 25
        rsi_extreme_overbought = rsi_7[i] > 75
        
        # Z-score signals (statistical extremes)
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        zscore_extreme_oversold = zscore[i] < -2.0
        zscore_extreme_overbought = zscore[i] > 2.0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        vol_normal = vol_ratio[i] > 0.8
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Z-score extreme oversold + RSI oversold + 1h bull trend
        if zscore_extreme_oversold and rsi_oversold and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: Z-score oversold + RSI oversold + volume spike
        elif zscore_oversold and rsi_oversold and vol_spike:
            new_signal = SIZE_BASE
        # Tertiary: RSI extreme oversold + volume normal + price > 1h HMA
        elif rsi_extreme_oversold and vol_normal and bull_trend:
            new_signal = SIZE_BASE
        # Quaternary: Z-score oversold + EMA bullish (looser for more trades)
        elif zscore_oversold and ema_bullish:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Z-score extreme overbought + RSI overbought + 1h bear trend
        if zscore_extreme_overbought and rsi_overbought and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: Z-score overbought + RSI overbought + volume spike
        elif zscore_overbought and rsi_overbought and vol_spike:
            new_signal = -SIZE_BASE
        # Tertiary: RSI extreme overbought + volume normal + price < 1h HMA
        elif rsi_extreme_overbought and vol_normal and bear_trend:
            new_signal = -SIZE_BASE
        # Quaternary: Z-score overbought + EMA bearish (looser for more trades)
        elif zscore_overbought and ema_bearish:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals