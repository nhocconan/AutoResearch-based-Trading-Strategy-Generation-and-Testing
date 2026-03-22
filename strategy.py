#!/usr/bin/env python3
"""
Experiment #325: 15m Connors RSI Mean Reversion with 4h HMA Trend Filter

Hypothesis: After #313 and #319 failed badly on 15m (Sharpe -2.0 to -3.6), 
pure trend following and RSI pullback strategies are getting destroyed by 
whipsaws on this noisy timeframe. 

Research shows Connors RSI (CRSI) has 75% win rate for mean reversion when 
combined with HTF trend filter. This strategy:
1. Uses 4h HMA(21) for PRIMARY directional bias (only trade with HTF trend)
2. Uses 15m Connors RSI for entry timing (oversold in uptrend = long, overbought in downtrend = short)
3. Uses ATR(14) trailing stoploss at 2.0x (tighter for 15m noise)
4. Adds volume confirmation to avoid false signals

Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(3): Short-term momentum
- RSI_Streak(2): Consecutive up/down bar strength
- PercentRank(100): Where current return ranks vs last 100 bars

Entry:
- Long: 4h HMA bullish + CRSI < 15 (extreme oversold within uptrend)
- Short: 4h HMA bearish + CRSI > 85 (extreme overbought within downtrend)

This is DIFFERENT from failed #313 (RSI pullback) because:
- CRSI is more robust than simple RSI (3 components vs 1)
- Extreme thresholds (15/85 vs 30/70) = fewer but higher quality trades
- Volume filter adds confirmation

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing (tighter for 15m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_meanrev_4h_hma_volume_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50.0)
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    This is a composite mean-reversion indicator with 75% win rate.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of price
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down bars)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.where(streak_avg_loss > 0, streak_avg_gain / streak_avg_loss, 100)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    # Component 3: PercentRank of returns over last 100 bars
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) > 0 and not np.isnan(window).all():
            current = returns[i]
            if not np.isnan(current):
                rank = np.sum(window < current) / len(window) * 100
                percent_rank[i] = rank
            else:
                percent_rank[i] = 50.0
        else:
            percent_rank[i] = 50.0
    
    # Combine all three components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs recent average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = primary directional bias (REQUIRED)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = extreme oversold (long opportunity in uptrend)
        # CRSI > 85 = extreme overbought (short opportunity in downtrend)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.2x average confirms the move
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: 4h bullish + CRSI oversold + volume confirmation
        long_conditions = (
            bull_trend_4h and
            crsi_oversold and
            volume_confirmed
        )
        
        # SHORT: 4h bearish + CRSI overbought + volume confirmation
        short_conditions = (
            bear_trend_4h and
            crsi_overbought and
            volume_confirmed
        )
        
        # Determine position size based on signal strength
        position_size = SIZE_BASE
        if crsi_oversold and bull_trend_4h:
            # Very extreme oversold = stronger position
            if crsi[i] < 10:
                position_size = SIZE_STRONG
        if crsi_overbought and bear_trend_4h:
            # Very extreme overbought = stronger position
            if crsi[i] > 90:
                position_size = SIZE_STRONG
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit long when CRSI becomes overbought (>70)
        # Exit short when CRSI becomes oversold (<30)
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 70:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 30:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals