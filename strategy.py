# 1d Weekly Pivot + Volume + Trend Filter
# Hypothesis: Fade at weekly S3/R3 in direction of weekly EMA(20) trend on daily chart
# with volume confirmation. Works in bull/bear by trading with weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_weekly_pivot_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivots and EMA trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivots from previous week
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Previous week's data
    prev_close = np.roll(close_weekly, 1)
    prev_high = np.roll(high_weekly, 1)
    prev_low = np.roll(low_weekly, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Weekly Camarilla levels (S3/R3)
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align weekly levels to daily
    R3_daily = align_htf_to_ltf(prices, df_weekly, R3)
    S3_daily = align_htf_to_ltf(prices, df_weekly, S3)
    
    # Weekly EMA(20) for trend filter
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_daily = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Volume filter: daily volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if required data not available
        if (np.isnan(R3_daily[i]) or np.isnan(S3_daily[i]) or 
            np.isnan(ema_20_daily[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 or trend changes
            if low[i] <= S3_daily[i] or close[i] < ema_20_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches R3 or trend changes
            if high[i] >= R3_daily[i] or close[i] > ema_20_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3 in direction of weekly EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_daily[i]:  # Uptrend
                    if low[i] <= S3_daily[i] and close[i] > S3_daily[i]:  # Bounce off S3
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if high[i] >= R3_daily[i] and close[i] < R3_daily[i]:  # Rejection at R3
                        position = -1
                        signals[i] = -0.25
    
    return signals