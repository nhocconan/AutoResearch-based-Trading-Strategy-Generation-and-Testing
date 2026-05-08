# 6H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# 6H timeframe with 1D trend filter (EMA34) and volume spike confirmation
# Long when price breaks above R3 + price above daily EMA34 + volume > 1.5x 20-period average
# Short when price breaks below S3 + price below daily EMA34 + volume > 1.5x 20-period average
# Exit when price crosses opposite S1/R1 level
# Designed for 60-100 total trades over 4 years (15-25/year) to minimize fee drag

name = "6H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots, EMA trend, and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily average volume for volume filter
    volume_daily = df_daily['volume'].values
    vol_ma_20_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla pivot levels
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Pivot point
    pivot = (high_daily + low_daily + close_daily) / 3.0
    
    # Camarilla levels
    daily_range = high_daily - low_daily
    r3 = pivot + daily_range * 1.1 / 4.0
    s3 = pivot - daily_range * 1.1 / 4.0
    r1 = pivot + daily_range * 1.1 / 12.0
    s1 = pivot - daily_range * 1.1 / 12.0
    
    # Align all daily indicators to 6h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20_daily)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period average of daily volume
        vol_filter = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for breakout with volume confirmation and trend alignment
            # Long: price breaks above R3 + price above daily EMA34 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_aligned[i]:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S3 + price below daily EMA34 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_aligned[i]:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below S1 level
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1 level
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals