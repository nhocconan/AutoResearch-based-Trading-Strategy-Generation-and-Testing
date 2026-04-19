# 12h_Pivot_R1S1_Breakout_Volume_Trend_v1
# 12h timeframe with 1d Camarilla pivot breakout, volume confirmation, and 1d EMA34 trend filter
# Strategy: Long when price breaks above R1 with volume > 1.5x average and price > EMA34
# Short when price breaks below S1 with volume > 1.5x average and price < EMA34
# Exit on opposite pivot level touch (S1 for long, R1 for short) or trend reversal
# Designed for low-frequency, high-conviction trades in both bull and bear markets
# Target: 15-35 trades/year to minimize fee drag

name = "12h_Pivot_R1S1_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot, volume average, and EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 1d average volume
        # Scale 1d average to 12h by dividing by 2 (since 2x12h = 1d)
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 2)
        
        if position == 0:
            # Look for long entry: price breaks above R1 + volume + uptrend (price > EMA34)
            if close[i] > r1_aligned[i] and volume_filter and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below S1 + volume + downtrend (price < EMA34)
            elif close[i] < s1_aligned[i] and volume_filter and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on touch of S1 or trend reversal
            if close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on touch of R1 or trend reversal
            if close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals