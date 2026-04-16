#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d Bollinger Bands (20, 2)
    close_1d_series = pd.Series(close_1d)
    ma_20 = close_1d_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d_series.rolling(window=20, min_periods=20).std().values
    bb_upper = ma_20 + 2 * std_20
    bb_lower = ma_20 - 2 * std_20
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # === 1w data (HTF for trend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 1w EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily indicators for entry timing ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma_10
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below BB lower OR RSI becomes overbought
            if (price < bb_lower_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above BB upper OR RSI becomes oversold
            if (price > bb_upper_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above BB upper AND above EMA34 (trend filter) 
                # AND RSI not overbought AND volume spike AND volatility not too high
                if (price > bb_upper_val) and (price > ema_34_1w_val) and (rsi_val < 60) and \
                   (vol_ratio_val > 2.0) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below BB lower AND below EMA34 (trend filter) 
                # AND RSI not oversold AND volume spike AND volatility not too high
                elif (price < bb_lower_val) and (price < ema_34_1w_val) and (rsi_val > 40) and \
                     (vol_ratio_val > 2.0) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 80)):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Bands_EMA34_RSI_Volume"
timeframe = "1d"
leverage = 1.0