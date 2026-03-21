#!/usr/bin/env python3
"""
Experiment #013: 15m Bollinger Regime + 4h Trend Filter + RSI/MACD Entries
Hypothesis: 15m timeframe captures intraday moves while 4h HTF filters major trend.
Bollinger BandWidth percentile detects regime (squeeze=range, expand=trend).
In trending regime: follow 4h HMA direction with 15m MACD/RSI pullback entries.
In ranging regime: RSI mean reversion at Bollinger band extremes.
ATR-based stoploss at 2.5x protects against crashes. Position size capped at 0.30.
This should generate 50-100 trades/year with adaptive logic for bull/bear/range.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_bbw_regime_15m_v1"
timeframe = "15m"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and BandWidth."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    return upper, lower, sma, bandwidth

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / std
    zscore = np.nan_to_num(zscore, nan=0.0)
    return zscore

def calculate_percentile_rank(values, window=100):
    """Calculate rolling percentile rank."""
    result = np.zeros(len(values))
    for i in range(window, len(values)):
        window_vals = values[i-window:i]
        result[i] = np.sum(window_vals < values[i]) / window
    return result

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger(close, 20, 2.0)
    zscore = calculate_zscore(close, 20)
    
    # Calculate Bollinger BandWidth percentile for regime detection
    bb_percentile = calculate_percentile_rank(bb_bandwidth, 100)
    
    # Volume SMA
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
        # 4h trend filter
        hma_4h_val = hma_4h_aligned[i]
        hma_4h_prev = hma_4h_aligned[i-1] if i > 0 else hma_4h_val
        
        # Determine 4h trend direction
        trend_4h_bullish = hma_4h_val > 0 and close[i] > hma_4h_val
        trend_4h_bearish = hma_4h_val > 0 and close[i] < hma_4h_val
        
        # Regime detection: BBWidth percentile
        # Low percentile (<30) = squeeze/range, High percentile (>70) = trend/expansion
        is_trending_regime = bb_percentile[i] > 0.40
        is_ranging_regime = bb_percentile[i] < 0.40
        
        new_signal = 0.0
        
        # ===== TRENDING REGIME LOGIC =====
        if is_trending_regime:
            # Long: 4h bullish + MACD bullish + RSI not overbought
            if trend_4h_bullish:
                macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] * 0.9
                rsi_ok = rsi[i] > 40 and rsi[i] < 75
                vol_ok = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
                
                if macd_bullish and rsi_ok and vol_ok:
                    new_signal = SIZE
            
            # Short: 4h bearish + MACD bearish + RSI not oversold
            elif trend_4h_bearish:
                macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] * 0.9
                rsi_ok = rsi[i] < 60 and rsi[i] > 25
                vol_ok = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
                
                if macd_bearish and rsi_ok and vol_ok:
                    new_signal = -SIZE
        
        # ===== RANGING REGIME LOGIC (Mean Reversion) =====
        elif is_ranging_regime:
            # Long: price at lower band + RSI oversold + zscore low
            at_lower_band = close[i] <= bb_lower[i] * 1.002
            rsi_oversold = rsi[i] < 45
            zscore_low = zscore[i] < -0.5
            
            if at_lower_band and rsi_oversold and zscore_low:
                new_signal = SIZE
            
            # Short: price at upper band + RSI overbought + zscore high
            at_upper_band = close[i] >= bb_upper[i] * 0.998
            rsi_overbought = rsi[i] > 55
            zscore_high = zscore[i] > 0.5
            
            if at_upper_band and rsi_overbought and zscore_high:
                new_signal = -SIZE
        
        # ===== STOPLOSS LOGIC (Rule 6) =====
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take partial profit at 3R
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if signals[i-1] == SIZE and new_signal == SIZE:
                    new_signal = HALF_SIZE  # Reduce to half
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take partial profit at 3R
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if signals[i-1] == -SIZE and new_signal == -SIZE:
                    new_signal = -HALF_SIZE  # Reduce to half
        
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