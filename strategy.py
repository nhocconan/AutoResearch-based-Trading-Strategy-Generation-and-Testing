#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation.
# Uses daily trend (price above/below daily Kijun Sen) to filter counter-trend trades,
# Ichimoku TK cross on 6h for entry timing, and volume to reduce false signals.
# Ichimoku provides inherent trend/momentum/cloud support/resistance, effective in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee impact.

name = "6h_ichimoku1d_vol_v1"
timeframe = "6h"
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
    
    # 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = np.full(n, np.nan)
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = np.full(n, np.nan)
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    
    # Calculate Tenkan-sen (9-period)
    for i in range(8, n):
        tenkan_sen[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Calculate Kijun-sen (26-period)
    for i in range(25, n):
        kijun_sen[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Calculate Senkou Span B (52-period) - needed for cloud
    senkou_span_b = np.full(n, np.nan)
    for i in range(51, n):
        senkou_span_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # 1-day trend filter: price vs daily Kijun Sen (26-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Kijun-sen (26-period)
    kijun_sen_1d = np.full(len(close_1d), np.nan)
    for i in range(25, len(close_1d)):
        kijun_sen_1d[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Align daily Kijun-sen to 6h timeframe
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need at least 52 for Senkou Span B)
    start = max(52, 26, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_b[i]) or np.isnan(kijun_sen_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Senkou Span A for current Ichimoku cloud
        # Senkou Span A = (Tenkan-sen + Kijun-sen) / 2
        senkou_span_a = (tenkan_sen[i] + kijun_sen[i]) / 2
        
        # Determine cloud boundaries (Senkou Span A and B shifted 26 periods ahead)
        # For current price, cloud is formed by Senkou Span A and B from 26 periods ago
        if i >= 26:
            senkou_span_a_cloud = (tenkan_sen[i-26] + kijun_sen[i-26]) / 2
            senkou_span_b_cloud = senkou_span_b[i-26]
            # Cloud top is max of Span A and B, bottom is min
            cloud_top = max(senkou_span_a_cloud, senkou_span_b_cloud)
            cloud_bottom = min(senkou_span_a_cloud, senkou_span_b_cloud)
        else:
            # Not enough data for cloud projection
            cloud_top = senkou_span_a
            cloud_bottom = senkou_span_a
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: TK cross down (Tenkan < Kijun) or price below cloud or stoploss
            if (tenkan_sen[i] < kijun_sen[i] or
                close[i] < cloud_bottom or
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i]):  # Simple volatility stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross up (Tenkan > Kijun) or price above cloud or stoploss
            if (tenkan_sen[i] > kijun_sen[i] or
                close[i] > cloud_top or
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: TK cross up (Tenkan > Kijun), price above cloud, volume, and above daily Kijun (bullish trend)
            if (tenkan_sen[i] > kijun_sen[i] and
                close[i] > cloud_top and
                volume_filter and
                close[i] > kijun_sen_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: TK cross down (Tenkan < Kijun), price below cloud, volume, and below daily Kijun (bearish trend)
            elif (tenkan_sen[i] < kijun_sen[i] and
                  close[i] < cloud_bottom and
                  volume_filter and
                  close[i] < kijun_sen_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-day Supertrend trend filter and volume confirmation.
# Uses daily Supertrend to filter counter-trend trades, 6h Donchian breakouts for entries,
# and volume to reduce false signals. Supertrend adapts to volatility, effective in both bull/bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee impact.

name = "6h_donchian20_1d_supertrend_vol_v1"
timeframe = "6h"
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
    
    # 6h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # 1-day Supertrend (ATR=10, multiplier=3)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) for 1d
    atr_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 11:
        tr_1d = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        if len(tr_1d) > 0:
            atr_1d[10] = np.mean(tr_1d[:10])
            for i in range(11, len(close_1d)):
                atr_1d[i] = (atr_1d[i-1] * 9 + tr_1d[i-1]) / 10
    
    # Supertrend calculation
    upper_band = np.full(len(close_1d), np.nan)
    lower_band = np.full(len(close_1d), np.nan)
    supertrend = np.full(len(close_1d), np.nan)
    trend = np.full(len(close_1d), np.nan)  # 1 for uptrend, -1 for downtrend
    
    if len(close_1d) >= 11 and not np.isnan(atr_1d[10]):
        for i in range(10, len(close_1d)):
            if np.isnan(atr_1d[i]):
                continue
            upper_band[i] = (high_1d[i] + low_1d[i]) / 2 + 3 * atr_1d[i]
            lower_band[i] = (high_1d[i] + low_1d[i]) / 2 - 3 * atr_1d[i]
            
            if i == 10:
                supertrend[i] = upper_band[i]
                trend[i] = 1
            else:
                if (supertrend[i-1] == upper_band[i-1] and close_1d[i] <= upper_band[i]) or \
                   (supertrend[i-1] == lower_band[i-1] and close_1d[i] >= lower_band[i]):
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
    
    # Align 1d Supertrend and trend to 6h
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 11)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: close below Supertrend or stoploss hit
            if (close[i] < supertrend_aligned[i] or
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: close above Supertrend or stoploss hit
            if (close[i] > supertrend_aligned[i] or
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above Donchian high with volume and uptrend (Supertrend up)
            if (close[i] > donch_high[i] and volume_filter and 
                trend_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with volume and downtrend (Supertrend down)
            elif (close[i] < donch_low[i] and volume_filter and 
                  trend_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from 1-day timeframe with volume confirmation.
# Uses daily Camarilla levels (H4, L4, H3, L3) for fade/breakout signals:
# - Fade at H3/L3 (revert to H4/L4) when price touches these levels with volume
# - Breakout continuation at H4/L4 when price breaks with volume and daily trend
# Daily trend filter (price vs daily 20-period EMA) avoids counter-trend trades.
# Camarilla levels work well in ranging markets (common in 2025+ BTC/ETH).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee impact.

name = "6h_camarilla1d_vol_v1"
timeframe = "6h"
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
    
    # 1-day Camarilla levels (based on previous day's range)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1]):
            continue
        diff = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i-1] + 1.5 * diff
        camarilla_l4[i] = close_1d[i-1] - 1.5 * diff
        camarilla_h3[i] = close_1d[i-1] + 1.125 * diff
        camarilla_l3[i] = close_1d[i-1] - 1.125 * diff
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Daily trend filter: 20-period EMA on 1-day close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.4x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.4
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches L4 (target) or stoploss hit
            if (close[i] <= camarilla_l4_aligned[i] or
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches H4 (target) or stoploss hit
            if (close[i] >= camarilla_h4_aligned[i] or
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long fade: price touches L3 with volume and above daily EMA (bullish bias)
            if (abs(close[i] - camarilla_l3_aligned[i]) < 0.001 * close[i] and  # Within 0.1% of L3
                volume_filter and
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short fade: price touches H3 with volume and below daily EMA (bearish bias)
            elif (abs(close[i] - camarilla_h3_aligned[i]) < 0.001 * close[i] and  # Within 0.1% of H3
                  volume_filter and
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            # Long breakout: price breaks above H4 with volume and above daily EMA
            elif (close[i] > camarilla_h4_aligned[i] and
                  volume_filter and
                  close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short breakout: price breaks below L4 with volume and below daily EMA
            elif (close[i] < camarilla_l4_aligned[i] and
                  volume_filter and
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-week EMA trend filter and volume confirmation.
# Uses weekly EMA(20) to determine primary trend (avoid counter-trend trades),
# Elder Ray to measure bull/bear power relative to EMA13 for entry timing,
# and volume to confirm strength of moves.
# Elder Ray identifies momentum shifts, effective in both trending and ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee impact.

name = "6h_elderray1w_ema_vol_v1"
timeframe = "6h"
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
    
    # 6h Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    # EMA13 for Elder Ray calculation
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2 + ema13[i-1] * 11) / 13
    
    bull_power = np.full(n, np.nan)
    bear_power = np.full(n, np.nan)
    if n >= 13:
        bull_power = high - ema13
        bear_power = low - ema13
    
    # 1-week EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_20w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20w[i] = (close_1w[i] * 2 + ema_20w[i-1] * 18) / 20
    
    # Align weekly EMA to 6h timeframe
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 13)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_20w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power turns positive (bulls weakening) or stoploss hit
            if (bear_power[i] > 0 or
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power turns negative (bears weakening) or stoploss hit
            if (bull_power[i] < 0 or
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: Bull Power > 0 and increasing (bulls gaining strength) with volume and above weekly EMA (bullish trend)
            if (bull_power[i] > 0 and
                i > start and bull_power[i] > bull_power[i-1] and  # Increasing bull power
                volume_filter and
                close[i] > ema_20w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power < 0 and decreasing (bears gaining strength) with volume and below weekly EMA (bearish trend)
            elif (bear_power[i] < 0 and
                  i > start and bear_power[i] < bear_power[i-1] and  # Decreasing bear power (more negative)
                  volume_filter and
                  close[i] < ema_20w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX(14) + Williams Alligator strategy with 1-day EMA filter.
# Uses ADX > 25 to identify trending markets, Williams Alligator (JAWS/TEETH/LIPS) for entry signals,
# and daily EMA(50) trend filter to avoid counter-trend trades.
# ADX filters out ranging markets, Alligator catches trends early.
# Effective in both bull and bear markets by only trading when ADX confirms trend strength.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee impact.

name = "6h_adx_alligator1d_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h ADX(14) calculation
    # +DI, -DI, DX
    tr = np.maximum(
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    )
    atr = np.full(n, np.nan)
    if len(tr) > 0 and n >= 15:
        atr[14] = np.mean(tr[:14])
        for i in range(15, n):
            atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # +DM and -DM
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    tr_smooth = np.full(n, np.nan)
    if n >= 14:
        plus_dm_smooth[13] = np.sum(plus_dm[1:14])
        minus_dm_smooth[13] = np.sum(minus_dm[1:14])
        tr_smooth[13] = np.sum(tr[1:14])
        for i in range(14, n):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
    
    # +DI, -DI, DX, ADX
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    if n >= 14 and tr_smooth[13] > 0:
        plus_di[13] = (plus_dm_smooth[13] / tr_smooth[13]) * 100
        minus_di[13] = (minus_dm_smooth[13] / tr_smooth[13]) * 100
        dx[13] = (np.abs(plus_di[13] - minus_di[13]) / (plus_di[13] + minus_di[13])) * 100
        for i in range(14, n):
            if tr_smooth[i] > 0:
                plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX is smoothed DX
    if n >= 28:
        adx[27] = np.mean(dx[14:28])
        for i in range(28, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # 6h Williams Alligator
    # JAWS (Blue): 13-period SMMA, smoothed 8 bars ahead
    # TEETH (Red): 8-period SMMA, smoothed 5 bars ahead
    # LIPS (Green): 5-period SMMA, smoothed 3 bars ahead
    # SMMA = Smoothed Moving Average (similar to EMA but different smoothing)
    def smma(arr, period):
        result