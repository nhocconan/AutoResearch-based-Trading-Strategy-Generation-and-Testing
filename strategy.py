#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h trend filter and volume confirmation
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA(50) AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND Bull Power < 0 AND price < 12h EMA(50) AND volume > 1.5x 20-period average
# Exit when Bull Power and Bear Power both become negative (bull exhaustion) or both positive (bear exhaustion)
# Uses 6h timeframe to balance trade frequency, 12h EMA for trend filter, Elder Ray for bull/bear power
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 6h performance

name = "6h_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Index components
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    df_12h_close = df_12h['close'].values
    df_12h_close_series = pd.Series(df_12h_close)
    df_12h_ema = df_12h_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    df_12h_ema_aligned = align_htf_to_ltf(prices, df_12h, df_12h_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(df_12h_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: exhaustion signals
        if position == 1:  # long position
            if bull_power[i] <= 0 and bear_power[i] >= 0:  # bull exhaustion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bull_power[i] >= 0 and bear_power[i] <= 0:  # bear exhaustion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA AND volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > df_12h_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 AND price < 12h EMA AND volume confirmation
            elif (bull_power[i] < 0 and bear_power[i] < 0 and 
                  close[i] < df_12h_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with volume confirmation
# Long when price breaks above R4 with volume > 2x 20-period average
# Short when price breaks below S4 with volume > 2x 20-period average
# Exit when price returns to the 1d pivot point (mean reversion to equilibrium)
# Uses Camarilla pivot levels derived from prior 1d OHLC to identify institutional support/resistance
# Volume confirms institutional participation in breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 6h performance

name = "6h_camarilla_pivot_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day OHLC data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1000)
    # S4 = C - ((H - L) * 1.1000)
    # R3 = C + ((H - L) * 1.1000/2)
    # S3 = C - ((H - L) * 1.1000/2)
    
    # Extract prior day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    camarilla_r4 = prev_close + (camarilla_range * 1.1000)
    camarilla_s4 = prev_close - (camarilla_range * 1.1000)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to pivot point (mean reversion)
        if position == 1:  # long position
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price breaks above R4 with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and close[i-1] <= camarilla_r4_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and close[i-1] >= camarilla_s4_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Long when Tenkan > Kijun AND price > Cloud AND price > 1d EMA(50) AND volume > 1.5x 20-period average
# Short when Tenkan < Kijun AND price < Cloud AND price < 1d EMA(50) AND volume > 1.5x 20-period average
# Exit when Tenkan and Kijun cross in opposite direction
# Uses Ichimoku Cloud for dynamic support/resistance, 1d EMA for higher timeframe trend filter
# Volume confirms institutional participation in trend continuation
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 6h performance

name = "6h_ichimoku_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    tenkan = (high_series.rolling(window=9, min_periods=9).max() + 
              low_series.rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (high_series.rolling(window=26, min_periods=26).max() + 
             low_series.rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = (high_series.rolling(window=52, min_periods=52).max() + 
                low_series.rolling(window=52, min_periods=52).min()) / 2
    
    # 1-day EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1d_close = df_1d['close'].values
    df_1d_close_series = pd.Series(df_1d_close)
    df_1d_ema = df_1d_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    df_1d_ema_aligned = align_htf_to_ltf(prices, df_1d, df_1d_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Senkou B lookback
        # Skip if required data not available
        if (np.isnan(tenkan.iloc[i]) or np.isnan(kijun.iloc[i]) or 
            np.isnan(senkou_a.iloc[i]) or np.isnan(senkou_b.iloc[i]) or 
            np.isnan(df_1d_ema_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B shifted forward 26 periods)
        # For cloud at current period, we need values from 26 periods ago
        if i >= 26:
            senkou_a_shifted = senkou_a.iloc[i-26]
            senkou_b_shifted = senkou_b.iloc[i-26]
            # Cloud top is max of Senkou A and B, cloud bottom is min
            cloud_top = max(senkou_a_shifted, senkou_b_shifted)
            cloud_bottom = min(senkou_a_shifted, senkou_b_shifted)
            
            tenkan_val = tenkan.iloc[i]
            kijun_val = kijun.iloc[i]
            close_val = close[i]
        else:
            # Not enough data for cloud calculation
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Tenkan and Kijun cross in opposite direction
        if position == 1:  # long position
            if tenkan_val < kijun_val:  # Death cross
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if tenkan_val > kijun_val:  # Golden cross
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Tenkan > Kijun AND price > Cloud AND price > 1d EMA AND volume confirmation
            if (tenkan_val > kijun_val and close_val > cloud_top and 
                close_val > df_1d_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price < Cloud AND price < 1d EMA AND volume confirmation
            elif (tenkan_val < kijun_val and close_val < cloud_bottom and 
                  close_val < df_1d_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Long when Williams %R crosses above -20 from below AND price > 1d EMA(50) AND volume > 1.5x 20-period average
# Short when Williams %R crosses below -80 from above AND price < 1d EMA(50) AND volume > 1.5x 20-period average
# Exit when Williams %R returns to -50 (mean reversion to midpoint)
# Williams %R identifies overbought/oversold levels, 1d EMA provides trend filter
# Volume confirms institutional participation in reversal moves
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 6h performance

name = "6h_williamsr_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # 1-day EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1d_close = df_1d['close'].values
    df_1d_close_series = pd.Series(df_1d_close)
    df_1d_ema = df_1d_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    df_1d_ema_aligned = align_htf_to_ltf(prices, df_1d, df_1d_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(williams_r.iloc[i]) or np.isnan(df_1d_ema_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        williams_r_val = williams_r.iloc[i]
        williams_r_prev = williams_r.iloc[i-1] if i > 0 else williams_r_val
        
        # Check exits: Williams %R returns to -50 (mean reversion)
        if position == 1:  # long position
            if williams_r_val >= -50:  # Returned to midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r_val <= -50:  # Returned to midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Williams %R crosses above -20 from below AND price > 1d EMA AND volume confirmation
            if (williams_r_val > -20 and williams_r_prev <= -20 and 
                close[i] > df_1d_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above AND price < 1d EMA AND volume confirmation
            elif (williams_r_val < -80 and williams_r_prev >= -80 and 
                  close[i] < df_1d_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

--- END OF FILE ---