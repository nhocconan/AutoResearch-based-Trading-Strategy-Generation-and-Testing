#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
# - Long when price breaks above 20-day high AND 1w EMA200 rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-day low AND 1w EMA200 falling AND volume > 1.5x 20-bar avg
# - Exit with ATR-based trailing stop: signal=0 when long and price < highest_high - 2*ATR(14) or short and price < lowest_low + 2*ATR(14)
# - Uses 1w EMA200 for strong trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian breakouts work well in trending markets; 1w EMA200 filter avoids whipsaws in ranging/bear markets

name = "1d_donchian_breakout_1w_ema_volume_v1"
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
    
    # Pre-compute 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute Donchian channels (20-period) from daily data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR(14) for trailing stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Track highest high since entry for trailing stop (long)
    highest_since_entry = np.full(n, np.nan)
    # Track lowest low since entry for trailing stop (short)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above 20-day high AND 1w uptrend with volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                prices['close'].iloc[i] > ema200_1w_aligned[i] and  # price above 1w EMA200
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = prices['high'].iloc[i]  # initialize tracking
            # Short when price breaks below 20-day low AND 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  prices['close'].iloc[i] < ema200_1w_aligned[i] and  # price below 1w EMA200
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
                lowest_since_entry[i] = prices['low'].iloc[i]  # initialize tracking
            else:
                signals[i] = 0.0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
        else:  # Have position - trail stop and check for exit
            # Update tracking levels
            if position == 1:  # Long position
                highest_since_entry[i] = max(highest_since_entry[i-1], prices['high'].iloc[i])
                # Exit when price drops below highest_high - 2*ATR (trailing stop)
                if prices['close'].iloc[i] < highest_since_entry[i] - 2.0 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                    highest_since_entry[i] = np.nan
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                lowest_since_entry[i] = min(lowest_since_entry[i-1], prices['low'].iloc[i])
                # Exit when price rises above lowest_low + 2*ATR (trailing stop)
                if prices['close'].iloc[i] > lowest_since_entry[i] + 2.0 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                    lowest_since_entry[i] = np.nan
                else:
                    signals[i] = -0.25
    
    return signals