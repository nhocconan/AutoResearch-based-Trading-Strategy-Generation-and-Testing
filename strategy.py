#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_1d_Trend_Signal"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(high_1d), np.nan)
    
    # Williams Fractal: bearish = high[n] is highest of [n-2, n-1, n, n+1, n+2]
    #                 bullish = low[n] is lowest of [n-2, n-1, n, n+1, n+2]
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals need 2 extra daily bars for confirmation (center bar + 2 more to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA34 for trend filter (needs only completed daily candle)
    ema_34_1d = pd.Series(close_1d:=df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6-period RSI for entry timing (on 6h timeframe)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_34_aligned[i]
        rsi_val = rsi[i]
        bullish_fractal = bullish_fractal_aligned[i]
        bearish_fractal = bearish_fractal_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: bullish fractal confirmed + price above EMA34 + RSI not overbought
            if (not np.isnan(bullish_fractal) and price > ema and 
                rsi_val < 70 and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal confirmed + price below EMA34 + RSI not oversold
            elif (not np.isnan(bearish_fractal) and price < ema and 
                  rsi_val > 30 and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below EMA34 or bearish fractal forms
            if price < ema or not np.isnan(bearish_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above EMA34 or bullish fractal forms
            if price > ema or not np.isnan(bullish_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals