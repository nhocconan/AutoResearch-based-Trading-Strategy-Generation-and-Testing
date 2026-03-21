#!/usr/bin/env python3
"""
Experiment #004: 4h KAMA + 1d HMA Trend Filter + Volatility Targeting
Hypothesis: 4h primary TF with 1d HTF trend filter. KAMA adapts to volatility
(reduces whipsaw in range markets vs EMA). Volatility targeting reduces size
in high-vol periods (protects during 2022 crash). Asymmetric sizing: bigger
longs in bull regime, smaller shorts in bear. Should generate 20-40 trades/year
with better risk-adjusted returns than pure trend following.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_vol_target_4h_v1"
timeframe = "4h"
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
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    In trending markets, KAMA follows price closely. In choppy markets, KAMA flattens.
    """
    n = len(close)
    close = np.array(close, dtype=float)
    
    # Efficiency Ratio (ER): measures trend strength
    signal = np.abs(close - np.roll(close, er_period))
    signal[:er_period] = np.abs(close[:er_period] - close[0])
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = np.abs(close[i] - close[i-1])
    
    # Sum of absolute price changes over ER period
    noise_sum = np.zeros(n)
    for i in range(er_period, n):
        noise_sum[i] = np.sum(np.abs(np.diff(close[max(0,i-er_period):i+1])))
    noise_sum[:er_period] = noise_sum[er_period] if n > er_period else 1.0
    
    er = np.zeros(n)
    mask = noise_sum > 0
    er[mask] = signal[mask] / noise_sum[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for trend direction."""
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
    return rsi

def calculate_volatility_regime(close, period=20):
    """
    Calculate volatility regime using rolling std percentile.
    Returns 1 for high vol, 0 for normal, -1 for low vol.
    """
    n = len(close)
    returns = np.diff(close, prepend=close[0]) / close
    rolling_std = pd.Series(returns).rolling(window=period, min_periods=period).std().values
    
    # Calculate percentile over last 100 bars
    vol_percentile = np.zeros(n)
    for i in range(period + 50, n):
        vol_percentile[i] = np.percentile(rolling_std[max(0,i-100):i+1], 70)
    
    regime = np.zeros(n)
    regime[rolling_std > vol_percentile * 1.3] = 1   # High vol
    regime[rolling_std < vol_percentile * 0.7] = -1  # Low vol
    # else normal (0)
    
    return regime, rolling_std

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, 14)
    vol_regime, vol_std = calculate_volatility_regime(close, 20)
    
    # Calculate KAMA slope for momentum
    kama_slope = np.zeros(n)
    for i in range(5, n):
        kama_slope[i] = (kama[i] - kama[i-5]) / kama[i-5] * 100
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    LOW_VOL_SIZE = 0.35  # Increase size in low vol
    HIGH_VOL_SIZE = 0.15  # Reduce size in high vol
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(50, n):
        # 1d trend filter (very slow, stable)
        hma_1d_val = hma_1d_aligned[i]
        hma_1d_prev = hma_1d_aligned[i-1] if i > 0 else hma_1d_val
        daily_trend = 1 if hma_1d_val > hma_1d_prev else -1
        
        # 4h KAMA trend
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # KAMA slope momentum
        kama_momentum = kama_slope[i]
        
        # RSI condition (relaxed for more trades)
        rsi_long = rsi[i] > 35  # Not extremely oversold
        rsi_short = rsi[i] < 65  # Not extremely overbought
        
        # Volatility regime sizing
        if vol_regime[i] == 1:
            size_mult = HIGH_VOL_SIZE / BASE_SIZE
        elif vol_regime[i] == -1:
            size_mult = LOW_VOL_SIZE / BASE_SIZE
        else:
            size_mult = 1.0
        
        # Entry logic (relaxed conditions for more trades)
        new_signal = 0.0
        
        # Long entry: daily uptrend + 4h bullish + RSI not oversold
        if daily_trend > 0 and kama_trend > 0 and rsi_long:
            new_signal = BASE_SIZE * size_mult
        
        # Short entry: daily downtrend + 4h bearish + RSI not overbought
        # Smaller shorts (asymmetric) - shorts are riskier in crypto
        elif daily_trend < 0 and kama_trend < 0 and rsi_short:
            new_signal = -BASE_SIZE * size_mult * 0.7  # 30% smaller shorts
        
        # Stoploss logic (Rule 6) - ATR-based
        if position_side > 0 and entry_price[i-1] > 0:
            stoploss_level = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stoploss_level:
                new_signal = 0.0  # Stoploss hit
            
            # Trailing stop for longs
            if close[i] > highest_since_entry[i-1]:
                highest_since_entry[i] = close[i]
                trail_stop = highest_since_entry[i] - 2.0 * atr[i]
                if close[i] > entry_price[i-1] + 2.0 * atr[i] and close[i] < trail_stop + atr[i]:
                    new_signal = HALF_SIZE * size_mult  # Take partial profit
        
        if position_side < 0 and entry_price[i-1] > 0:
            stoploss_level = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stoploss_level:
                new_signal = 0.0  # Stoploss hit
            
            # Trailing stop for shorts
            if close[i] < lowest_since_entry[i-1]:
                lowest_since_entry[i] = close[i]
                trail_stop = lowest_since_entry[i] + 2.0 * atr[i]
                if close[i] < entry_price[i-1] - 2.0 * atr[i] and close[i] > trail_stop - atr[i]:
                    new_signal = -HALF_SIZE * size_mult  # Take partial profit
        
        # Update position tracking
        if i > 0:
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
                entry_price[i] = entry_price[i-1]
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
                if position_side != 0 and new_signal == 0:
                    position_side = 0  # Position closed
        else:
            entry_price[i] = close[i] if new_signal != 0 else 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        # Ensure signal is within bounds (Rule 4)
        new_signal = np.clip(new_signal, -0.40, 0.40)
        
        # Round to discrete levels to reduce fee churn
        if abs(new_signal) < 0.10:
            new_signal = 0.0
        elif new_signal > 0:
            if new_signal >= 0.30:
                new_signal = 0.30
            elif new_signal >= 0.15:
                new_signal = 0.15
            else:
                new_signal = 0.15
        else:
            if new_signal <= -0.30:
                new_signal = -0.30
            elif new_signal <= -0.15:
                new_signal = -0.15
            else:
                new_signal = -0.15
        
        signals[i] = new_signal
    
    return signals