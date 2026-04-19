#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volume regime
# - Use 4h EMA(20) as trend filter (bullish if close > EMA20)
# - Use 1d volume regime: high volume days (>1.5x 20-day avg) allow entries, low volume days restrict to counter-trend
# - On 1h: enter long when RSI(14) crosses above 30 in bullish trend OR below 70 in bearish trend (mean reversion in ranges)
# - Enter short when RSI(14) crosses below 70 in bullish trend OR above 30 in bearish trend
# - Exit when RSI returns to 50 (mean) or trend changes
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Position size: 0.20 (20%)
# - Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag
# - Works in bull markets via trend-following RSI pulls backs, in bear via mean reversion in ranges

name = "1h_EMA20_RSI_VolumeRegime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False).values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for volume regime
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate RSI(14) on 1h data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
            
        # Determine trend from 4h EMA
        bullish_trend = close[i] > ema_4h_aligned[i]
        
        # Volume regime: 1 = high volume (allow trend following), 0 = low volume (mean reversion only)
        vol_ratio = vol_ma_1d_aligned[i]
        high_volume_regime = vol_ratio > 0 and volume[i] > 1.5 * (vol_ratio / 24)  # Scale 1d avg to 1h
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Look for long entry
            if bullish_trend:
                # In bullish trend: buy RSI pullback above 30
                if (rsi_values[i] > 30 and rsi_values[i-1] <= 30 and 
                    (high_volume_regime or True)):  # Allow in both regimes but stronger in high vol
                    signals[i] = 0.20
                    position = 1
            else:
                # In bearish trend: mean reversion - buy RSI bounce above 30 only in low volume
                if (rsi_values[i] > 30 and rsi_values[i-1] <= 30 and 
                    not high_volume_regime):
                    signals[i] = 0.20
                    position = 1
                    
            # Look for short entry
            if bullish_trend:
                # In bullish trend: mean reversion - sell RSI rejection below 70 only in low volume
                if (rsi_values[i] < 70 and rsi_values[i-1] >= 70 and 
                    not high_volume_regime):
                    signals[i] = -0.20
                    position = -1
            else:
                # In bearish trend: sell RSI rejection below 70
                if (rsi_values[i] < 70 and rsi_values[i-1] >= 70 and 
                    (high_volume_regime or True)):
                    signals[i] = -0.20
                    position = -1
                    
        elif position == 1:
            # Long position: exit when RSI returns to 50 or trend changes
            if (rsi_values[i] >= 50 and rsi_values[i-1] < 50) or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit when RSI returns to 50 or trend changes
            if (rsi_values[i] <= 50 and rsi_values[i-1] > 50) or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals