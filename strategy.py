# 4h_camarilla_pivot_12h_volume_v1
# Hypothesis: Camarilla pivot levels from 12h timeframe identify key support/resistance levels
# where price is likely to reverse or break. Combined with volume confirmation on 4h and
# a 12h trend filter, this strategy captures breakouts with institutional significance.
# Works in both bull and bear markets by following the higher timeframe trend.
# Target: 20-40 trades/year to minimize fee drag on 4h timeframe.
name = "4h_camarilla_pivot_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for Camarilla pivots and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar
    # Based on previous 12h bar's high, low, close
    ph = df_12h['high'].values
    pl = df_12h['low'].values
    pc = df_12h['close'].values
    
    # Camarilla levels (using previous bar's data)
    # Resistance levels
    r4 = pc + ((ph - pl) * 1.5000)
    r3 = pc + ((ph - pl) * 1.2500)
    r2 = pc + ((ph - pl) * 1.1666)
    r1 = pc + ((ph - pl) * 1.0833)
    # Support levels
    s1 = pc - ((ph - pl) * 1.0833)
    s2 = pc - ((ph - pl) * 1.1666)
    s3 = pc - ((ph - pl) * 1.2500)
    s4 = pc - ((ph - pl) * 1.5000)
    
    # Pivot point
    pivot = (ph + pl + pc) / 3
    
    # Align all levels to 4h time
    r4_4h = align_htf_to_ltf(prices, df_12h, r4)
    r3_4h = align_htf_to_ltf(prices, df_12h, r3)
    r2_4h = align_htf_to_ltf(prices, df_12h, r2)
    r1_4h = align_htf_to_ltf(prices, df_12h, r1)
    pivot_4h = align_htf_to_ltf(prices, df_12h, pivot)
    s1_4h = align_htf_to_ltf(prices, df_12h, s1)
    s2_4h = align_htf_to_ltf(prices, df_12h, s2)
    s3_4h = align_htf_to_ltf(prices, df_12h, s3)
    s4_4h = align_htf_to_ltf(prices, df_12h, s4)
    
    # 12h EMA(21) for trend filter
    ema_12h = pd.Series(pc).ewm(span=21, adjust=False).mean().values
    ema_12h_4h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_4h[i]) or np.isnan(r3_4h[i]) or np.isnan(r2_4h[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(pivot_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(s2_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(ema_12h_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below pivot or 12h trend turns bearish
            if close[i] < pivot_4h[i] or close[i] < ema_12h_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above pivot or 12h trend turns bullish
            if close[i] > pivot_4h[i] or close[i] > ema_12h_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_ok = volume[i] > 1.5 * vol_ma[i]
            
            # Breakout above R3 with volume and bullish 12h trend
            if close[i] > r3_4h[i] and vol_ok and close[i] > ema_12h_4h[i]:
                position = 1
                signals[i] = 0.25
            # Breakdown below S3 with volume and bearish 12h trend
            elif close[i] < s3_4h[i] and vol_ok and close[i] < ema_12h_4h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals