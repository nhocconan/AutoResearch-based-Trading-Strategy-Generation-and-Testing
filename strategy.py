#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) during 12h uptrend.
# Short when Williams %R crosses below -80 (overbought rejection) during 12h downtrend.
# Uses volume > 1.3x 20-period average for confirmation.
# 12h trend filter avoids counter-trend trades. Williams %R captures short-term reversals.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 12h trend filter: EMA(21) slope
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_slope = np.diff(ema_12h, prepend=ema_12h[0])
    trend_up = ema_slope > 0
    trend_down = ema_slope < 0
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if trend data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Williams %R cross conditions
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_curr = williams_r[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R crosses below -80 or trend turns down
            if (wr_prev > -80 and wr_curr <= -80) or trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -20 or trend turns up
            if (wr_prev < -20 and wr_curr >= -20) or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: Williams %R crosses above -20 during uptrend
                if (wr_prev <= -20 and wr_curr > -20) and trend_up_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Williams %R crosses below -80 during downtrend
                elif (wr_prev >= -80 and wr_curr < -80) and trend_down_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) during 12h uptrend.
# Short when Williams %R crosses below -80 (overbought rejection) during 12h downtrend.
# Uses volume > 1.3x 20-period average for confirmation.
# 12h trend filter avoids counter-trend trades. Williams %R captures short-term reversals.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 12h trend filter: EMA(21) slope
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_slope = np.diff(ema_12h, prepend=ema_12h[0])
    trend_up = ema_slope > 0
    trend_down = ema_slope < 0
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if trend data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Williams %R cross conditions
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_curr = williams_r[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R crosses below -80 or trend turns down
            if (wr_prev > -80 and wr_curr <= -80) or trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -20 or trend turns up
            if (wr_prev < -20 and wr_curr >= -20) or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: Williams %R crosses above -20 during uptrend
                if (wr_prev <= -20 and wr_curr > -20) and trend_up_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Williams %R crosses below -80 during downtrend
                elif (wr_prev >= -80 and wr_curr < -80) and trend_down_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) during 12h uptrend.
# Short when Williams %R crosses below -80 (overbought rejection) during 12h downtrend.
# Uses volume > 1.3x 20-period average for confirmation.
# 12h trend filter avoids counter-trend trades. Williams %R captures short-term reversals.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 12h trend filter: EMA(21) slope
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_slope = np.diff(ema_12h, prepend=ema_12h[0])
    trend_up = ema_slope > 0
    trend_down = ema_slope < 0
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if trend data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Williams %R cross conditions
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_curr = williams_r[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R crosses below -80 or trend turns down
            if (wr_prev > -80 and wr_curr <= -80) or trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -20 or trend turns up
            if (wr_prev < -20 and wr_curr >= -20) or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: Williams %R crosses above -20 during uptrend
                if (wr_prev <= -20 and wr_curr > -20) and trend_up_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Williams %R crosses below -80 during downtrend
                elif (wr_prev >= -80 and wr_curr < -80) and trend_down_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) during 12h uptrend.
# Short when Williams %R crosses below -80 (overbought rejection) during 12h downtrend.
# Uses volume > 1.3x 20-period average for confirmation.
# 12h trend filter avoids counter-trend trades. Williams %R captures short-term reversals.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 12h trend filter: EMA(21) slope
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_slope = np.diff(ema_12h, prepend=ema_12h[0])
    trend_up = ema_slope > 0
    trend_down = ema_slope < 0
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if trend data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Williams %R cross conditions
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_curr = williams_r[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R crosses below -80 or trend turns down
            if (wr_prev > -80 and wr_curr <= -80) or trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -20 or trend turns up
            if (wr_prev < -20 and wr_curr >= -20) or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: Williams %R crosses above -20 during uptrend
                if (wr_prev <= -20 and wr_curr > -20) and trend_up_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Williams %R crosses below -80 during downtrend
                elif (wr_prev >= -80 and wr_curr < -80) and trend_down_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) during 12h uptrend.
# Short when Williams %R crosses below -80 (overbought rejection) during 12h downtrend.
# Uses volume > 1.3x 20-period average for confirmation.
# 12h trend filter avoids counter-trend trades. Williams %R captures short-term reversals.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 12h trend filter: EMA(21) slope
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_slope = np.diff(ema_12h, prepend=ema_12h[0])
    trend_up = ema_slope > 0
    trend_down = ema_slope < 0
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if trend data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Williams %R cross conditions
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_curr = williams_r[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R crosses below -80 or trend turns down
            if (wr_prev > -80 and wr_curr <= -80) or trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -20 or trend turns up
            if (wr_prev < -20 and wr_curr >= -20) or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: Williams %R crosses above -20 during uptrend
                if (wr_prev <= -20 and wr_curr > -20) and trend_up_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Williams %R crosses below -80 during downtrend
                elif (wr_prev >= -80 and wr_curr < -80) and trend_down_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) during 12h uptrend.
# Short when Williams %R crosses below -80 (overbought rejection) during 12h downtrend.
# Uses volume > 1.3x 20-period average for confirmation.
# 12h trend filter avoids counter-trend trades. Williams %R captures short-term reversals.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 12h trend filter: EMA(21) slope
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_slope = np.diff(ema_12h, prepend=ema_12h[0])
    trend_up = ema_slope > 0
    trend_down = ema_slope < 0
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if trend data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Williams %R cross conditions
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_curr = williams_r[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R crosses below -80 or trend turns down
            if (wr_prev > -80 and wr_curr <= -80) or trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -20 or trend turns up
            if (wr_prev < -20 and wr_curr >= -20) or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: Williams %R crosses above -20 during uptrend
                if (wr_prev <= -20 and wr_curr > -20) and trend_up_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Williams %R crosses below -80 during downtrend
                elif (wr_prev >= -80 and wr_curr < -80) and trend_down_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) during 12h uptrend.
# Short when Williams %R crosses below -80 (overbought rejection) during 12h downtrend.
# Uses volume > 1.3x 20-period average for confirmation.
# 12h trend filter avoids counter-trend trades. Williams %R captures short-term reversals.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 12h trend filter: EMA(21) slope
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_slope = np.diff(ema_12h, prepend=ema_12h[0])
    trend_up = ema_slope > 0
    trend_down = ema_slope < 0
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if trend data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Williams %R cross conditions
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_curr = williams_r[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R crosses below -80 or trend turns down
            if (wr_prev > -80 and wr_curr <= -80) or trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -20 or trend turns up
            if (wr_prev < -20 and wr_curr >= -20) or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter: