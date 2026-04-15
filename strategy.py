# Solution: 1d_Kelly_WeeklyTrend
# Hypothesis: On daily timeframe, use weekly trend (EMA50) for bias, and enter on pullbacks with Kelly sizing based on weekly volatility.
# In bull markets, weekly trend up → buy pullbacks; in bear markets, weekly trend down → sell rallies.
# Kelly sizing reduces exposure in high volatility (bear markets) and increases in low volatility (bull markets), managing drawdowns.
# Uses only 2 conditions: weekly trend + price position relative to weekly EMA.
# Expects ~10-20 trades/year, avoiding fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data once
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly close array
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA(50) for trend
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Weekly volatility (ATR-like: std dev of returns) for Kelly scaling
    returns_weekly = np.diff(np.log(close_weekly))
    vol_weekly = pd.Series(returns_weekly).rolling(window=20, min_periods=20).std().values
    # Prepend first value to match length
    vol_weekly = np.concatenate([np.full(20, np.nan), vol_weekly])
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Get aligned weekly indicators
        ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)[i]
        vol_aligned = align_htf_to_ltf(prices, df_weekly, vol_weekly)[i]
        
        # Check for NaN values
        if np.isnan(ema50_aligned) or np.isnan(vol_aligned):
            continue
        
        # Weekly trend filter
        if close[i] > ema50_aligned:
            # Bullish bias: buy on dip (price below weekly EMA but above prior low)
            if close[i] < ema50_aligned and low[i] < low[i-1]:  # simple pullback signal
                # Kelly fraction: assume edge = 0.02, vol = vol_aligned, kelly = edge/vol^2
                edge = 0.02
                if vol_aligned > 0:
                    kelly = edge / (vol_aligned ** 2)
                    # Cap Kelly at 0.3 and scale by trend strength
                    trend_strength = min((close[i] - ema50_aligned) / ema50_aligned, 0.1)  # max 10% above
                    size = 0.3 * (kelly / 0.5) * (1 + trend_strength)  # normalize kelly
                    size = max(0.05, min(size, 0.3))  # clamp between 5% and 30%
                    if position <= 0:
                        position = 1
                        signals[i] = size
                else:
                    if position <= 0:
                        position = 1
                        signals[i] = 0.15  # fallback size
            # Exit: close above weekly EMA with strength
            elif position == 1 and close[i] > ema50_aligned * 1.02:
                position = 0
                signals[i] = 0.0
        else:
            # Bearish bias: sell on rally (price above weekly EMA but below prior high)
            if close[i] > ema50_aligned and high[i] > high[i-1]:  # simple rally signal
                edge = 0.02
                if vol_aligned > 0:
                    kelly = edge / (vol_aligned ** 2)
                    trend_strength = min((ema50_aligned - close[i]) / ema50_aligned, 0.1)
                    size = 0.3 * (kelly / 0.5) * (1 + trend_strength)
                    size = max(0.05, min(size, 0.3))
                    if position >= 0:
                        position = -1
                        signals[i] = -size
                else:
                    if position >= 0:
                        position = -1
                        signals[i] = -0.15
            # Exit: close below weekly EMA with strength
            elif position == -1 and close[i] < ema50_aligned * 0.98:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_Kelly_WeeklyTrend"
timeframe = "1d"
leverage = 1.0