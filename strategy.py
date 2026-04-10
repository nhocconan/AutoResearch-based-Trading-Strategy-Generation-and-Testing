#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d volume spike
# - Entry: Long when 1h RSI < 30 + price > 4h EMA50 + 1d volume > 2.0x 20-period average
#          Short when 1h RSI > 70 + price < 4h EMA50 + 1d volume > 2.0x 20-period average
# - Exit: Close-based reversal - exit long when RSI > 50, exit short when RSI < 50
# - Position sizing: 0.20 (discrete levels to minimize fee churn)
# - Uses 1h for precise timing, 4h EMA for trend direction, 1d volume for conviction
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Target: 15-35 trades/year (60-140 total over 4 years) to stay within HARD MAX: 200 total
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)

name = "1h_4h_1d_rsi_meanrev_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # Pre-compute 4h close for EMA
    close_4h = df_4h['close'].values
    
    # Pre-compute 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # Calculate 1h RSI (14-period)
    delta = pd.Series(close_1h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA (50-period)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 1h timeframe
    rsi_1h_aligned = rsi_1h  # already 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if outside trading session or any required data is invalid
        if not in_session[i] or \
           np.isnan(rsi_1h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or \
           np.isnan(volume_ma_aligned[i]) or np.isnan(volume_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get current values
        rsi = rsi_1h_aligned[i]
        ema_50 = ema_50_4h_aligned[i]
        close_price = close_1h[i]
        volume = volume_1d_aligned[i]
        volume_ma = volume_ma_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmation = volume > 2.0 * volume_ma
        
        if position == 0:  # Flat - look for new entries
            # Long entry: RSI < 30 (oversold) + price above 4h EMA50 (uptrend) + volume confirmation
            if (rsi < 30.0 and 
                close_price > ema_50 and 
                volume_confirmation):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 (overbought) + price below 4h EMA50 (downtrend) + volume confirmation
            elif (rsi > 70.0 and 
                  close_price < ema_50 and 
                  volume_confirmation):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when RSI > 50 (momentum fading)
            # Exit short when RSI < 50 (momentum fading)
            if position == 1:
                if rsi > 50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi < 50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals