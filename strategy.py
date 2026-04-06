#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout with Volume Confirmation and Weekly EMA Trend Filter
# Long when price breaks above 20-day Donchian high with volume > 1.5x 20-day average and weekly EMA > prior weekly EMA
# Short when price breaks below 20-day Donchian low with volume > 1.5x 20-day average and weekly EMA < prior weekly EMA
# Uses daily close for breakout and weekly EMA for trend filter to work in both bull and bear markets
# Target: 30-100 trades over 4 years (7-25/year)

name = "1d_donchian20_weekly_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily 20-period Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Daily 20-period volume average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Weekly EMA trend filter (21-period)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        if i == 20:
            ema_1w[i] = np.mean(close_1w[:21])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (21 + 1)) + ema_1w[i-1] * (20 / (21 + 1))
    
    # Align weekly EMA to daily timeframe (shifted by 1 weekly bar)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Weekly EMA trend (previous week's EMA for trend direction)
        ema_prev = ema_1w_aligned[i-1] if i > 0 else ema_1w_aligned[i]
        ema_rising = ema_1w_aligned[i] > ema_prev
        ema_falling = ema_1w_aligned[i] < ema_prev
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to midpoint of Donchian channel
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to midpoint of Donchian channel
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: breakout above Donchian high with rising weekly EMA
                if close[i] > donchian_high[i] and ema_rising:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low with falling weekly EMA
                elif close[i] < donchian_low[i] and ema_falling:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout with Volume Confirmation and Weekly EMA Trend Filter
# Long when price breaks above 20-day Donchian high with volume > 1.5x 20-day average and weekly EMA > prior weekly EMA
# Short when price breaks below 20-day Donchian low with volume > 1.5x 20-day average and weekly EMA < prior weekly EMA
# Uses daily close for breakout and weekly EMA for trend filter to work in both bull and bear markets
# Target: 30-100 trades over 4 years (7-25/year)

name = "1d_donchian20_weekly_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily 20-period Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Daily 20-period volume average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Weekly EMA trend filter (21-period)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        if i == 20:
            ema_1w[i] = np.mean(close_1w[:21])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (21 + 1)) + ema_1w[i-1] * (20 / (21 + 1))
    
    # Align weekly EMA to daily timeframe (shifted by 1 weekly bar)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Weekly EMA trend (previous week's EMA for trend direction)
        ema_prev = ema_1w_aligned[i-1] if i > 0 else ema_1w_aligned[i]
        ema_rising = ema_1w_aligned[i] > ema_prev
        ema_falling = ema_1w_aligned[i] < ema_prev
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to midpoint of Donchian channel
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to midpoint of Donchian channel
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: breakout above Donchian high with rising weekly EMA
                if close[i] > donchian_high[i] and ema_rising:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low with falling weekly EMA
                elif close[i] < donchian_low[i] and ema_falling:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals