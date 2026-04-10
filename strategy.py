#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Camarilla pivot levels on 1h: breakout above H3 = long, below L3 = short
# - Volume confirmation: current 4h volume > 1.3x 20-period EMA (avoid low-volume fakeouts)
# - Trend filter: price > 1d EMA(50) for longs, price < 1d EMA(50) for shorts
# - Exit: opposite Camarilla breakout (H3/L3 reversal) or time-based exit (12h max hold)
# - Position sizing: 0.20 discrete level
# - Session filter: 08-20 UTC to avoid Asian session noise
# - Targets ~15-35 trades/year on 1h timeframe. Uses Camarilla for precise entry/exit,
#   volume confirmation reduces whipsaws, 1d EMA ensures alignment with higher timeframe trend.
#   Works in bull/bear: breakouts capture momentum, volume/trend filters improve win rate.

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    close_1d = df_1d['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_in_trade = 0  # track time in trade for max hold
    
    # Calculate Camarilla pivot levels on 1h (using previous bar's OHLC)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Using previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_hl = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * range_hl * 1.1 / 4
    camarilla_l3 = prev_close - 1.1 * range_hl * 1.1 / 4
    
    # Calculate 4h volume EMA for confirmation
    volume_ema_20_4h = pd.Series(volume_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ema_20_4h)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            bars_in_trade = 0  # reset if forced flat
            continue
        
        # HTF volume confirmation: 4h volume > 1.3x its 20-period EMA
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)
        vol_confirm_4h = vol_4h_current[i] > 1.3 * volume_ema_20_4h_aligned[i]
        
        # Entry conditions
        long_entry = (close[i] > camarilla_h3[i] and 
                     vol_confirm_4h and 
                     close[i] > ema_50_1d_aligned[i])
        short_entry = (close[i] < camarilla_l3[i] and 
                      vol_confirm_4h and 
                      close[i] < ema_50_1d_aligned[i])
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.20
                bars_in_trade = 0
            elif short_entry:
                position = -1
                signals[i] = -0.20
                bars_in_trade = 0
            else:
                signals[i] = 0.0
                bars_in_trade = 0
        else:  # Have position - look for exit
            bars_in_trade += 1
            # Exit conditions: opposite Camarilla breakout or max hold (12 bars = 12h)
            if position == 1:  # Long position
                if (close[i] < camarilla_l3[i] or  # opposite breakout
                    bars_in_trade >= 12):          # max hold time
                    position = 0
                    signals[i] = 0.0
                    bars_in_trade = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if (close[i] > camarilla_h3[i] or  # opposite breakout
                    bars_in_trade >= 12):          # max hold time
                    position = 0
                    signals[i] = 0.0
                    bars_in_trade = 0
                else:
                    signals[i] = -0.20
    
    return signals