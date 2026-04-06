#!/usr/bin/env python3
"""
1h Donchian(20) Breakout + 4h EMA Trend + Volume Spike + 1d Trend Filter
Hypothesis: Uses 4h EMA for medium-term trend bias and 1d trend for higher timeframe filter,
combined with 1h Donchian breakouts for entry timing. Volume spike filters out low-quality breaks.
Designed for 15-30 trades/year to minimize fee drag while capturing trending moves.
Works in bull (long breakouts with uptrend) and bear (short breakdowns with downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian20_4hetma_1dtrend_vol_v1"
timeframe = "1h"
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
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 4h EMA20 for trend bias
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 20:
        ema_4h[19] = np.mean(close_4h[:20])
        for i in range(20, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_4h = np.where(close_4h > ema_4h, 1, -1)
    
    # Align 4h trend to 1h
    trend_bias_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_bias_4h)
    
    # 1d EMA50 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend bias: above EMA = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 1h
    trend_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_bias_4h_aligned[i]) or np.isnan(trend_bias_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR against 4h trend OR 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                trend_bias_4h_aligned[i] == -1 or
                trend_bias_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.20
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR against 4h trend OR 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                trend_bias_4h_aligned[i] == 1 or
                trend_bias_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.20
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + 4h trend + 1d trend + volume
            # Minimum 6 bars between trades to reduce churn
            if bars_since_exit >= 6:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with bullish 4h AND 1d trend and volume
                if bull_breakout and trend_bias_4h_aligned[i] == 1 and trend_bias_1d_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: bearish breakout with bearish 4h AND 1d trend and volume
                elif bear_breakout and trend_bias_4h_aligned[i] == -1 and trend_bias_1d_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Donchian(20) Breakout + 4h EMA Trend + Volume Spike + 1d Trend Filter
Hypothesis: Uses 4h EMA for medium-term trend bias and 1d trend for higher timeframe filter,
combined with 1h Donchian breakouts for entry timing. Volume spike filters out low-quality breaks.
Designed for 15-30 trades/year to minimize fee drag while capturing trending moves.
Works in bull (long breakouts with uptrend) and bear (short breakdowns with downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian20_4hetma_1dtrend_vol_v1"
timeframe = "1h"
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
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 4h EMA20 for trend bias
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 20:
        ema_4h[19] = np.mean(close_4h[:20])
        for i in range(20, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_4h = np.where(close_4h > ema_4h, 1, -1)
    
    # Align 4h trend to 1h
    trend_bias_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_bias_4h)
    
    # 1d EMA50 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend bias: above EMA = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 1h
    trend_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_bias_4h_aligned[i]) or np.isnan(trend_bias_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR against 4h trend OR 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                trend_bias_4h_aligned[i] == -1 or
                trend_bias_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.20
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR against 4h trend OR 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                trend_bias_4h_aligned[i] == 1 or
                trend_bias_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.20
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + 4h trend + 1d trend + volume
            # Minimum 6 bars between trades to reduce churn
            if bars_since_exit >= 6:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with bullish 4h AND 1d trend and volume
                if bull_breakout and trend_bias_4h_aligned[i] == 1 and trend_bias_1d_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: bearish breakout with bearish 4h AND 1d trend and volume
                elif bear_breakout and trend_bias_4h_aligned[i] == -1 and trend_bias_1d_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals

</think>