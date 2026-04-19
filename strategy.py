# 6h_RSI_Momentum_Volume_Breakout
# Uses RSI momentum + volume spike + 1-day trend filter for breakout entries
# Works in both bull and bear markets by capturing momentum bursts with confirmation
# Target: 20-30 trades/year per symbol
name = "6h_RSI_Momentum_Volume_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1-day average volume (20-period) for volume spike detection
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6-period RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 6-period price rate of change for momentum confirmation
    roc = ((close - np.roll(close, 6)) / np.roll(close, 6)) * 100
    roc[:6] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure EMA200 and ROC are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema200 = ema200_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        rsi_val = rsi[i]
        roc_val = roc[i]
        
        # Trend filter: only long in uptrend (price > EMA200), only short in downtrend (price < EMA200)
        uptrend = price > ema200
        downtrend = price < ema200
        
        if position == 0:
            # Long entry: RSI > 50 (bullish momentum) + ROC > 0 (positive momentum) + volume spike + uptrend
            if rsi_val > 50 and roc_val > 0 and vol > 2.0 * vol_ma and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI < 50 (bearish momentum) + ROC < 0 (negative momentum) + volume spike + downtrend
            elif rsi_val < 50 and roc_val < 0 and vol > 2.0 * vol_ma and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI < 40 (loss of momentum) or trend reversal
            if rsi_val < 40 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI > 60 (loss of bearish momentum) or trend reversal
            if rsi_val > 60 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals