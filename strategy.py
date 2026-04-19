# 12h_Camarilla_Pivot_Breakout_Volume_1wTrend_v1
# Hypothesis: 12h timeframe with daily Camarilla pivot breakout, weekly trend filter, and volume confirmation
# - Uses daily Camarilla pivot points (R1, S1) as key support/resistance levels
# - Weekly trend filter ensures we only trade in the direction of the higher timeframe trend
# - Volume confirmation requires current volume > 1.5x 20-period average for conviction
# - Designed to work in both bull and bear markets by following weekly trend direction
# - Target: 20-50 trades/year to minimize fee drag (12h timeframe naturally limits frequency)
# - Entry: Long when price breaks above daily R1 with weekly uptrend and volume confirmation
# - Entry: Short when price breaks below daily S1 with weekly downtrend and volume confirmation
# - Exit: Opposite pivot level touch (S1 for long, R1 for short) or weekly trend reversal
# - Position size: 0.25 (25%) to balance return and drawdown

name = "12h_Camarilla_Pivot_Breakout_Volume_1wTrend_v1"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot points for each daily bar
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    hl_range = (df_1d['high'] - df_1d['low']).values
    r1 = df_1d['close'].values + hl_range * 1.1 / 12
    s1 = df_1d['close'].values - hl_range * 1.1 / 12
    
    # Align daily pivot levels to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(34) for trend direction
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session filter (optional for 12h - can trade all hours)
    # 12h bars naturally cover full days, so session filter less critical
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: price breaks above daily R1 + weekly uptrend + volume
            if close[i] > r1_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below daily S1 + weekly downtrend + volume
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on touch of daily S1 or weekly trend reversal
            if close[i] < s1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on touch of daily R1 or weekly trend reversal
            if close[i] > r1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals