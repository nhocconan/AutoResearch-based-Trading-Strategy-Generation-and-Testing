#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = pd.to_datetime(prices['open_time'])
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = open_time.dt.hour.values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR (14-period) for volatility
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly ATR (14-period) for regime filter
    high_w = df_1d['high'].values
    low_w = df_1d['low'].values
    close_w = df_1d['close'].values
    tr1_w = high_w - low_w
    tr2_w = np.abs(high_w - np.roll(close_w, 1))
    tr3_w = np.abs(low_w - np.roll(close_w, 1))
    tr2_w[0] = np.inf
    tr3_w[0] = np.inf
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_w_avg = pd.Series(atr_w).rolling(window=50, min_periods=50).mean().values
    atr_w_ratio = atr_w / atr_w_avg
    atr_w_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_w_ratio)
    
    # Calculate daily EMA(50) for trend
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily RSI(14) for momentum
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).values
    
    # Calculate daily volume average
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(atr_w_ratio_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid low volatility periods (ATR ratio > 0.5%)
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005
        
        # Volume confirmation: current volume > 1.2x daily average
        vol_confirm = vol > (vol_avg_1d_aligned[i] * 1.2)
        
        # Trend filter: price relative to daily EMA50
        trend_long = price > ema_50_1d_aligned[i]
        trend_short = price < ema_50_1d_aligned[i]
        
        # Momentum filter: RSI in neutral zone (30-70)
        rsi_neutral = 30 <= rsi[i] <= 70
        
        # Regime filter: avoid extremely high volatility weeks (ATR ratio < 2.0)
        vol_regime = atr_w_ratio_aligned[i] < 2.0
        
        if position == 0:
            # Long setup: price above EMA50 + volume + volatility + RSI + regime
            if (trend_long and vol_confirm and vol_filter and rsi_neutral and vol_regime):
                position = 1
                signals[i] = position_size
            # Short setup: price below EMA50 + volume + volatility + RSI + regime
            elif (trend_short and vol_confirm and vol_filter and rsi_neutral and vol_regime):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50
            if price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA50
            if price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_EMA50_Volume_RSI_Regime"
timeframe = "1d"
leverage = 1.0