# 4h_1d_Camarilla_R1S1_Breakout_Volume
# Hypothesis: 4h timeframe with Camarilla pivot levels (R1/S1) from daily chart for breakout signals.
# Enters only during 08-20 UTC session with volume confirmation.
# Uses 1d Camarilla levels (R1/S1) as key support/resistance levels.
# Targets 20-50 trades/year (80-200 total over 4 years) with strict entry conditions.
# Works in bull/bear by following price action at institutional pivot levels.
# Camarilla levels are widely watched by institutions, providing high-probability breakout zones.

name = "4h_1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot levels (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (R1, S1)
    # Camarilla formulas: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = previous close, H = previous high, L = previous low
    prev_close = close_1d
    prev_high = high_1d
    prev_low = low_1d
    
    # Shift to get previous day's values (avoid look-ahead)
    prev_close_shifted = np.roll(prev_close, 1)
    prev_high_shifted = np.roll(prev_high, 1)
    prev_low_shifted = np.roll(prev_low, 1)
    # First day has no previous day, set to NaN
    prev_close_shifted[0] = np.nan
    prev_high_shifted[0] = np.nan
    prev_low_shifted[0] = np.nan
    
    # Calculate Camarilla R1 and S1 for previous day
    camarilla_R1 = prev_close_shifted + (prev_high_shifted - prev_low_shifted) * 1.1 / 12
    camarilla_S1 = prev_close_shifted - (prev_high_shifted - prev_low_shifted) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA and shifted values
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume
            if close[i] > camarilla_R1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume
            elif close[i] < camarilla_S1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Camarilla S1 (reversal signal)
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Camarilla R1 (reversal signal)
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals