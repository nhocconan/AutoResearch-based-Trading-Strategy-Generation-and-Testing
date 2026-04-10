#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-bar EMA pullback strategy with 4h volume confirmation and 1d trend filter
# - Entry: On 1h, wait for EMA(4) pullback during 4h-volume-confirmed breakouts
# - Long: Price > EMA(4) after touching EMA(4) from above during bullish 4h candle with volume confirmation
# - Short: Price < EMA(4) after touching EMA(4) from below during bearish 4h candle with volume confirmation
# - Trend filter: 1d EMA(50) alignment - only long when price > 1d EMA(50), short when price < 1d EMA(50)
# - Exit: Opposite EMA(4) touch or max 8-bar hold (8h) to prevent overstaying
# - Session filter: 08-20 UTC to reduce noise
# - Position size: 0.20 discrete
# - Designed for low trade frequency (target: 15-30/year) with high win rate by trading only
#   high-probability pullbacks in strong trends with volume confirmation
# - Works in bull/bear: captures momentum continuations while avoiding counter-trend traps

name = "1h_4h_1d_ema_pullback_v1"
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
    
    # Calculate 1h EMA(4) for entry signals
    close_s = pd.Series(close)
    ema_4 = close_s.ewm(span=4, min_periods=4, adjust=False).mean().values
    
    # Calculate 4h volume EMA for confirmation
    volume_ema_20_4h = pd.Series(volume_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ema_20_4h)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine 4h candle direction for volume confirmation context
    # Bullish 4h candle: close > open
    # Bearish 4h candle: close < open
    open_4h = df_4h['open'].values
    fourh_bullish = close_4h > open_4h
    fourh_bearish = close_4h < open_4h
    fourh_bullish_aligned = align_htf_to_ltf(prices, df_4h, fourh_bullish)
    fourh_bearish_aligned = align_htf_to_ltf(prices, df_4h, fourh_bearish)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Track EMA touch state for pullback detection
    ema_touch_long = np.zeros(n, dtype=bool)   # price touched EMA from above
    ema_touch_short = np.zeros(n, dtype=bool)  # price touched EMA from below
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_4[i]) or np.isnan(volume_ema_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            bars_in_trade = 0  # reset if forced flat
            ema_touch_long[i] = False
            ema_touch_short[i] = False
            continue
        
        # Update EMA touch states (using previous bar to avoid look-ahead)
        if i > 0:
            # Long touch: price was at or above EMA, now below EMA (pullback to EMA from above)
            ema_touch_long[i] = (close[i-1] >= ema_4[i-1]) and (close[i] < ema_4[i])
            # Short touch: price was at or below EMA, now above EMA (pullback to EMA from below)
            ema_touch_short[i] = (close[i-1] <= ema_4[i-1]) and (close[i] > ema_4[i])
        else:
            ema_touch_long[i] = False
            ema_touch_short[i] = False
        
        # HTF volume confirmation: 4h volume > 1.3x its 20-period EMA
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)
        vol_confirm_4h = vol_4h_current[i] > 1.3 * volume_ema_20_4h_aligned[i]
        
        # Entry conditions require EMA touch + volume confirmation + trend alignment
        long_entry = (ema_touch_long[i] and 
                     vol_confirm_4h and 
                     fourh_bullish_aligned[i] and  # 4h candle bullish
                     close[i] > ema_50_1d_aligned[i])  # above 1d EMA(50)
        short_entry = (ema_touch_short[i] and 
                      vol_confirm_4h and 
                      fourh_bearish_aligned[i] and  # 4h candle bearish
                      close[i] < ema_50_1d_aligned[i])  # below 1d EMA(50)
        
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
            # Exit conditions: opposite EMA touch or max hold (8 bars = 8h)
            if position == 1:  # Long position
                if (ema_touch_short[i] or  # opposite touch (price pulling back to EMA from below)
                    bars_in_trade >= 8):   # max hold time
                    position = 0
                    signals[i] = 0.0
                    bars_in_trade = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if (ema_touch_long[i] or   # opposite touch (price pulling back to EMA from above)
                    bars_in_trade >= 8):   # max hold time
                    position = 0
                    signals[i] = 0.0
                    bars_in_trade = 0
                else:
                    signals[i] = -0.20
    
    return signals