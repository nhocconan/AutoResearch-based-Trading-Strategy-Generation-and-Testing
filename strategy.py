#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
# - Primary timeframe: 1h for precise entry timing
# - 4h EMA(50) as trend filter: long when price > EMA50, short when price < EMA50
# - 1h RSI(14) < 30 for long entry, > 70 for short entry (mean reversion in trending market)
# - Volume confirmation: current 1h volume > 1.5x 20-period average
# - Session filter: 08-20 UTC to avoid low-liquidity periods
# - Fixed position size: 0.20 (20% of capital) to control risk and minimize fee churn
# - Designed for 1h timeframe: targets 15-37 trades/year to avoid fee drag
# - Works in bull/bear markets: 4h EMA filter ensures we trade with higher timeframe trend

name = "1h_4h_rsi_meanrev_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Pre-compute 1h RSI(14)
    close_1h = prices['close'].values
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute 1h volume confirmation
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1h > (1.5 * avg_volume_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(in_session[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or trend change
            if rsi[i] > 50 or close_1h[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or trend change
            if rsi[i] < 50 or close_1h[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for RSI mean reversion with trend and volume filters
            if vol_spike[i]:
                # Long: RSI oversold in uptrend
                if rsi[i] < 30 and close_1h[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: RSI overbought in downtrend
                elif rsi[i] > 70 and close_1h[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals