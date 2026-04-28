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
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss.rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe (this strategy uses 4h primary)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 4h Bollinger Bands for volatility regime
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + bb_std * std_bb
    lower_bb = sma_bb - bb_std * std_bb
    bb_width = (upper_bb - lower_bb) / sma_bb
    
    # Bollinger Band width percentile for regime detection (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) == 20 else np.nan, raw=True
    ).values
    
    # Regime: low volatility (squeeze) when BB width below 50th percentile
    low_vol_regime = bb_width < bb_width_percentile
    
    # Hour filter: 8-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(low_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        # Momentum filter: RSI in moderate range (avoid extremes)
        rsi_mid = 40 < rsi_aligned[i] < 60
        
        # Volatility filter: low volatility regime (squeeze)
        vol_filter = low_vol_regime[i]
        
        # Entry conditions - selective to reduce trades
        long_entry = trend_up and rsi_mid and vol_filter
        short_entry = trend_down and rsi_mid and vol_filter
        
        # Exit conditions: opposite trend or volatility expansion
        long_exit = not trend_up or not rsi_mid or not vol_filter
        short_exit = not trend_down or not rsi_mid or not vol_filter
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_EMA50_RSI_BB_Squeeze"
timeframe = "4h"
leverage = 1.0