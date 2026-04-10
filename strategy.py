#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(50) trend filter + volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w close > EMA50 AND volume > 1.5x avg
# - Short when price breaks below Donchian(20) low AND 1w close < EMA50 AND volume > 1.5x avg
# - Exit when price crosses Donchian(20) midpoint (mean reversion) or opposite breakout
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Targets ~10-20 trades/year (40-80 total over 4 years) to minimize fee drag
# - Donchian channels provide clear structure; 1w EMA ensures alignment with higher timeframe trend
# - Volume confirmation prevents false breakouts
# - Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with mean reversion exit)

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian(20) channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: max(high, lookback=20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: min(low, lookback=20)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: (donch_high + donch_low) / 2
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high AND 1w uptrend AND volume spike
            if (close[i] > donch_high[i] and 
                close[i] > ema50_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.30
            # Short signal: price breaks below Donchian low AND 1w downtrend AND volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midpoint (mean reversion)
            # 2. Opposite breakout (strong reversal signal)
            if position == 1:  # Long position
                if close[i] < donch_mid[i] or close[i] < donch_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30  # Hold long
            elif position == -1:  # Short position
                if close[i] > donch_mid[i] or close[i] > donch_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30  # Hold short
    
    return signals