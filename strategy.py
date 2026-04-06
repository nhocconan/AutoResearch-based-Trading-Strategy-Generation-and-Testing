#!/usr/bin/env python3
"""
1h Price Action with 4h Trend Filter and Volume Confirmation
Hypothesis: On 1h, take long when price closes above 4h VWAP with volume spike,
short when below 4h VWAP with volume spike. Use 1d trend filter to avoid counter-trend trades.
VWAP acts as dynamic support/resistance, volume confirms institutional interest.
Target: 80-150 total trades over 4 years (20-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwap_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # 1h VWAP (cumulative reset each day)
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_tpv, cum_vol, out=np.zeros_like(cum_tpv), where=cum_vol!=0)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h VWAP
    tpv_4h = ((high_4h + low_4h + close_4h) / 3.0) * volume_4h
    cum_tpv_4h = np.cumsum(tpv_4h)
    cum_vol_4h = np.cumsum(volume_4h)
    vwap_4h = np.divide(cum_tpv_4h, cum_vol_4h, out=np.zeros_like(cum_tpv_4h), where=cum_vol_4h!=0)
    
    # 1d trend filter: higher highs/lows (uptrend) or lower highs/lows (downtrend)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 20-period EMA on 1d for trend
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(100, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap[i]) or np.isnan(vwap_4h[i]) or np.isnan(trend_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below 1h VWAP OR trend turns down
            if (close[i] < vwap[i] or
                trend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above 1h VWAP OR trend turns up
            if (close[i] > vwap[i] or
                trend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: price closes above 1h VWAP with volume in uptrend
            if (close[i] > vwap[i] and
                trend_1d_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price closes below 1h VWAP with volume in downtrend
            elif (close[i] < vwap[i] and
                  trend_1d_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h VWAP Mean Reversion with 4h Trend Filter
Hypothesis: Price deviates from 1h VWAP then mean reverts. Enter when price
deviates >1.5*ATR from VWAP in direction of 4h trend. Volume confirms.
Targets 80-150 trades over 4 years by using strict deviation threshold.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwap_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price and VWAP (daily reset)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # Reset cumulative sums at midnight UTC (00:00)
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    vwap = np.full(n, np.nan)
    
    # Find daily reset points (00:00 UTC)
    # Assuming first bar starts at 00:00 or we don't have exact times
    # Simple approach: reset every 24 bars (24h / 1h = 24)
    # Better: use date change detection from open_time
    dates = pd.to_datetime(prices['open_time']).date
    date_changes = np.concatenate(([True], dates[1:] != dates[:-1]))
    
    for i in range(n):
        if date_changes[i]:
            cum_tpv[i] = tpv[i]
            cum_vol[i] = volume[i]
        else:
            cum_tpv[i] = cum_tpv[i-1] + tpv[i]
            cum_vol[i] = cum_vol[i-1] + volume[i]
        
        if cum_vol[i] > 0:
            vwap[i] = cum_tpv[i] / cum_vol[i]
    
    # ATR for deviation measurement
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 50-period EMA on 4h for trend
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 18) / 20
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Volume filter: current volume > 1.8x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(100, 20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(trend_4h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Price deviation from VWAP
        dev = close[i] - vwap[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to VWAP OR trend turns down
            if (dev <= 0 or
                trend_4h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price returns to VWAP OR trend turns up
            if (dev >= 0 or
                trend_4h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: price deviated from VWAP in trend direction
            # Long: price below VWAP in uptrend (mean reversion long)
            if (dev < 0 and
                abs(dev) > 1.5 * atr[i] and
                trend_4h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price above VWAP in downtrend (mean reversion short)
            elif (dev > 0 and
                  abs(dev) > 1.5 * atr[i] and
                  trend_4h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Dual Threshold Breakout with Volume and Trend Filter
Hypothesis: Breakouts require both price and volume confirmation.
Enter when price breaks ±0.5*ATR from VWAP AND volume > 2x average.
Use 4h trend filter to avoid counter-trend trades. Target: 90-160 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_dual_threshold_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price and VWAP (daily reset)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # Daily VWAP with reset at midnight UTC
    dates = pd.to_datetime(prices['open_time']).date
    date_changes = np.concatenate(([True], dates[1:] != dates[:-1]))
    
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    vwap = np.full(n, np.nan)
    
    for i in range(n):
        if date_changes[i]:
            cum_tpv[i] = tpv[i]
            cum_vol[i] = volume[i]
        else:
            cum_tpv[i] = cum_tpv[i-1] + tpv[i]
            cum_vol[i] = cum_vol[i-1] + volume[i]
        
        if cum_vol[i] > 0:
            vwap[i] = cum_tpv[i] / cum_vol[i]
    
    # ATR for volatility measurement
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 30-period EMA on 4h for trend (smoother)
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 30:
        ema_4h[29] = np.mean(close_4h[:30])
        for i in range(30, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 28) / 30
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Volume filter: current volume > 2.2x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(100, 20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(trend_4h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.2
        
        # Price deviation from VWAP in ATR units
        dev_atr = abs(close[i] - vwap[i]) / atr[i] if atr[i] > 0 else 0
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to VWAP OR trend turns down
            if (close[i] <= vwap[i] or
                trend_4h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price returns to VWAP OR trend turns up
            if (close[i] >= vwap[i] or
                trend_4h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: significant deviation with volume
            # Long: price above VWAP with volume in uptrend
            if (close[i] > vwap[i] and
                dev_atr > 0.5 and
                trend_4h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price below VWAP with volume in downtrend
            elif (close[i] < vwap[i] and
                  dev_atr > 0.5 and
                  trend_4h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Institutional Flow Detection with 4h Trend Filter
Hypothesis: Follow smart money - large volume moves in direction of higher timeframe trend.
Enter when volume spikes >3x average AND price moves >0.7*ATR from VWAP in trend direction.
Targets 70-130 trades by requiring strong volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_institutional_flow_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price and VWAP (daily reset)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # Daily VWAP with reset at midnight UTC
    dates = pd.to_datetime(prices['open_time']).date
    date_changes = np.concatenate(([True], dates[1:] != dates[:-1]))
    
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    vwap = np.full(n, np.nan)
    
    for i in range(n):
        if date_changes[i]:
            cum_tpv[i] = tpv[i]
            cum_vol[i] = volume[i]
        else:
            cum_tpv[i] = cum_tpv[i-1] + tpv[i]
            cum_vol[i] = cum_vol[i-1] + volume[i]
        
        if cum_vol[i] > 0:
            vwap[i] = cum_tpv[i] / cum_vol[i]
    
    # ATR for volatility measurement
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 50-period EMA on 4h for trend
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 18) / 20
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Volume filter: current volume > 3.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(100, 20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(trend_4h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition - require very strong volume
        volume_filter = volume[i] > vol_ma[i] * 3.0
        
        # Price deviation from VWAP in ATR units
        price_dev = abs(close[i] - vwap[i])
        dev_atr = price_dev / atr[i] if atr[i] > 0 else 0
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to VWAP OR trend turns down
            if (close[i] <= vwap[i] or
                trend_4h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price returns to VWAP OR trend turns up
            if (close[i] >= vwap[i] or
                trend_4h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: require both volume spike and price movement
            # Long: price above VWAP with volume spike in uptrend
            if (close[i] > vwap[i] and
                dev_atr > 0.7 and
                trend_4h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price below VWAP with volume spike in downtrend
            elif (close[i] < vwap[i] and
                  dev_atr > 0.7 and
                  trend_4h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Volume-Weighted Trend Following with 4h Filter
Hypothesis: Trend following works best when confirmed by institutional volume.
Use VWAP as dynamic trend indicator - price above VWAP = uptrend, below = downtrend.
Enter on pullbacks to VWAP in direction of 4h trend with volume confirmation.
Targets 85-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwap_trend_following_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price and VWAP (daily reset)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # Daily VWAP with reset at midnight UTC
    dates = pd.to_datetime(prices['open_time']).date
    date_changes = np.concatenate(([True], dates[1:] != dates[:-1]))
    
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    vwap = np.full(n, np.nan)
    
    for i in range(n):
        if date_changes[i]:
            cum_tpv[i] = tpv[i]
            cum_vol[i] = volume[i]
        else:
            cum_tpv[i] = cum_tpv[i-1] + tpv[i]
            cum_vol[i] = cum_vol[i-1] + volume[i]
        
        if cum_vol[i] > 0:
            vwap[i] = cum_tpv[i] / cum_vol[i]
    
    # ATR for pullback measurement
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 50-period EMA on 4h for trend
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 18) / 20
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Volume filter: current volume > 1.6x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(100, 20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(trend_4h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.6
        
        # Price position relative to VWAP
        above_vwap = close[i] > vwap[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below VWAP OR trend turns down
            if (not above_vwap or
                trend_4h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price crosses above VWAP OR trend turns up
            if (above_vwap or
                trend_4h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: pullbacks to VWAP in trend direction
            # Long: pullback to VWAP (price near VWAP from below) in uptrend
            if (close[i] <= vwap[i] * 1.005 and  # within 0.5% above VWAP
                close[i] >= vwap[i] * 0.995 and  # within 0.5% below VWAP
                close[i] > vwap[i] and           # but actually above VWAP for long
                trend_4h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: pullback to VWAP (price near VWAP from above) in downtrend
            elif (close[i] >= vwap[i] * 0.995 and  # within 0.5% below VWAP
                  close[i] <= vwap[i] * 1.005 and  # within 0.5% above VWAP
                  close[i] < vwap[i] and           # but actually below VWAP for short
                  trend_4h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Adaptive VWAP Bands with Trend Filter
Hypothesis: Price tends to revert to VWAP but trends persist. 
Enter when price touches VWAP bands (±1*ATR) in direction of 4h trend.
Volume confirms institutional participation. Targets 75-140 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwap_bands_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price and VWAP (daily reset)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # Daily VWAP with reset at midnight UTC
    dates = pd.to_datetime(prices['open_time']).date
    date_changes = np.concatenate(([True], dates[1:] != dates[:-1]))
    
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    vwap = np.full(n, np.nan)
    
    for i in range(n):
        if date_changes[i]:
            cum_tpv[i] = tpv[i]
            cum_vol[i] = volume[i]
        else:
            cum_tpv[i] = cum_tpv[i-1] + tpv[i]
            cum_vol[i] = cum_vol[i-1] + volume[i]
        
        if cum_vol[i] > 0:
            vwap[i] = cum_tpv[i] /