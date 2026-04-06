#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume and ADX Filter
Hypothesis: Breakouts from 20-period Donchian channels on 12h timeframe, filtered by volume and ADX, capture trends across market regimes.
Volume confirms breakout strength, ADX ensures trending conditions, and volatility-based stoploss manages risk.
Works in bull markets (long breakouts) and bear markets (short breakouts).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_adx"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # 12h ADX(14) for trend strength
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move = np.where(up_move > down_move, up_move, 0)
    down_move = np.where(down_move > up_move, down_move, 0)
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    up_ma = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    down_ma = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    # Directional Indicators
    plus_di = 100 * up_ma / (tr_ma + 1e-10)
    minus_di = 100 * down_ma / (tr_ma + 1e-10)
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian20 and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + ADX + trend filter
            bullish_breakout = close[i] > donchian_high[i-1]
            bearish_breakout = close[i] < donchian_low[i-1]
            strong_trend = adx[i] > 25
            uptrend_filter = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]  # Rising EMA
            downtrend_filter = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]  # Falling EMA
            
            if bullish_breakout and vol_filter[i] and strong_trend and uptrend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bearish_breakout and vol_filter[i] and strong_trend and downtrend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
12h ADX Trend with RSI Pullback Entry
Hypothesis: In trending markets (ADX > 25), pullbacks to the 21-period EMA on 12h timeframe offer high-probability entries.
RSI identifies overextended pullbacks, volume confirms participation, and volatility-based stops manage risk.
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_adx_trend_rsi_pullback"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA(21) for dynamic support/resistance
    ema_21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    # 12h RSI(14) for pullback identification
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)  # Slightly lower threshold for pullbacks
    
    # 12h ADX(14) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move = np.where(up_move > down_move, up_move, 0)
    down_move = np.where(down_move > up_move, down_move, 0)
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    up_ma = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    down_ma = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * up_ma / (tr_ma + 1e-10)
    minus_di = 100 * down_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA21 and ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_21[i]) or np.isnan(rsi[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI overbought OR price breaks below EMA21 OR stoploss
            if (rsi[i] > 70 or
                close[i] < ema_21[i] or
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI oversold OR price breaks above EMA21 OR stoploss
            if (rsi[i] < 30 or
                close[i] > ema_21[i] or
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: ADX trend + RSI pullback + volume
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]  # Rising daily EMA
            downtrend = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]  # Falling daily EMA
            strong_trend = adx[i] > 25
            pullback_long = (rsi[i] < 40 and close[i] <= ema_21[i])  # Oversold pullback to EMA
            pullback_short = (rsi[i] > 60 and close[i] >= ema_21[i])  # Overbought rally to EMA
            
            if uptrend and strong_trend and pullback_long and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif downtrend and strong_trend and pullback_short and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
12h Volume-Weighted Average Price (VWAP) Deviation with Trend Filter
Hypothesis: Price deviations from VWAP on 12h timeframe, filtered by higher timeframe trend and volume, offer mean-reversion opportunities in ranging markets and continuation in trending markets.
VWAP acts as dynamic support/resistance, volume confirms institutional interest, and ADX filters for appropriate market regime.
Works in bull markets (buy VWAP dips in uptrend) and bear markets (sell VWAP rallies in downtrend).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_vwap_deviation_trend_filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h VWAP calculation
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).cumsum().values
    vwap_denominator = pd.Series(volume).cumsum().values
    vwap = vwap_numerator / (vwap_denominator + 1e-10)
    
    # 12h standard deviation of price from VWAP (20-period)
    price_dev = close - vwap
    dev_ma = pd.Series(price_dev).rolling(window=20, min_periods=20).mean().values
    dev_std = pd.Series(price_dev).rolling(window=20, min_periods=20).std().values
    
    # 12h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # 12h ADX(14) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move = np.where(up_move > down_move, up_move, 0)
    down_move = np.where(down_move > up_move, down_move, 0)
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    up_ma = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    down_ma = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * up_ma / (tr_ma + 1e-10)
    minus_di = 100 * down_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For VWAP std and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(vwap[i]) or np.isnan(dev_std[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to VWAP OR stoploss
            if (close[i] >= vwap[i] or
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to VWAP OR stoploss
            if (close[i] <= vwap[i] or
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: VWAP deviation + volume + trend filter
            dev_zscore = price_dev[i] / (dev_std[i] + 1e-10)
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]  # Rising daily EMA
            downtrend = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]  # Falling daily EMA
            strong_trend = adx[i] > 25
            ranging = adx[i] < 20  # Weak trend = ranging market
            
            # In ranging markets: mean reversion at 2 standard deviations
            # In trending markets: continuation at 1 standard deviation
            if ranging and dev_zscore < -2.0 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif ranging and dev_zscore > 2.0 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            elif uptrend and strong_trend and dev_zscore < -1.0 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif downtrend and strong_trend and dev_zscore > 1.0 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
12h Hull Moving Average (HMA) Trend with Volume Confirmation
Hypothesis: Hull Moving Average reduces lag while maintaining smoothness, making it effective for trend identification on 12h timeframe.
Price crossing above/below HMA(21) signals trend changes, volume confirms institutional participation, and ADX filter avoids whipsaws.
Works in bull markets (buy HMA crosses up) and bear markets (sell HMA crosses down).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_hma_trend_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Hull Moving Average (HMA) calculation
    def hma(arr, period):
        """Calculate Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA of half period
        wma_half = pd.Series(arr).rolling(window=half_period, min_periods=half_period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
        ).values
        
        # WMA of full period
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
        ).values
        
        # Raw HMA: 2*WMA(half) - WMA(full)
        hma_raw = 2 * wma_half - wma_full
        
        # Final HMA: WMA of raw HMA with sqrt period
        hma_result = pd.Series(hma_raw).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
        ).values
        
        return hma_result
    
    # Calculate HMA(21)
    hma_21 = hma(close, 21)
    
    # 12h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # 12h ADX(14) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move = np.where(up_move > down_move, up_move, 0)
    down_move = np.where(down_move > up_move, down_move, 0)
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    up_ma = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    down_ma = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * up_ma / (tr_ma + 1e-10)
    minus_di = 100 * down_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For HMA calculation and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(hma_21[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below HMA OR stoploss
            if (close[i] < hma_21[i] or
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above HMA OR stoploss
            if (close[i] > hma_21[i] or
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: HMA cross + volume + trend filter
            hma_cross_up = close[i] > hma_21[i] and close[i-1] <= hma_21[i-1]
            hma_cross_down = close[i] < hma_21[i] and close[i-1] >= hma_21[i-1]
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]  # Rising daily EMA
            downtrend = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]  # Falling daily EMA
            strong_trend = adx[i] > 25
            
            if hma_cross_up and vol_filter[i] and strong_trend and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif hma_cross_down and vol_filter[i] and strong_trend and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
12h Elder Ray Index with Trend Filter
Hypothesis: Elder Ray Index measures bullish and bearish power behind price movements, identifying when trends are gaining or losing strength.
Combined with EMA trend filter and volume confirmation, it captures strong trend continuations while avoiding weak moves.
Works in bull markets (buy when bullish power increases in uptrend) and bear markets (sell when bearish power increases in downtrend).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_elder_ray_trend_filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # 12h Elder Ray Index components
    bull_power = high - ema_13  # Bullish power: high minus EMA
    bear_power = low - ema_13   # Bearish power: low minus EMA
    
    # 12h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # 12h ADX(14) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move = np.where(up_move > down_move, up_move, 0)
    down_move = np.where(down_move > up_move, down_move, 0)
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    up_ma = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    down_ma = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * up_ma / (tr_ma + 1e-10)
    minus_di = 100 * down_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA13 and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_13[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: bull power weakening OR stoploss
            if (bull_power[i] < bull_power[i-1] or
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bear power weakening OR stoploss
            if (bear_power[i] > bear_power[i-1] or
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals + volume + trend filter
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]  # Rising daily EMA
            downtrend = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]  # Falling daily EMA
            strong_tr