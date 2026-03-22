#!/usr/bin/env python3
"""
Experiment #597: 1h Multi-Signal Ensemble (Vol Spike + Mean Reversion + HTF Trend)

Hypothesis: After 596 experiments, the clearest pattern is that PURE trend strategies
fail on BTC/ETH (especially 2022 crash, 2025 bear). The winning approach combines:

1. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 indicates panic/extreme vol
   - After vol spike, price tends to revert (vol crush pattern)
   - Entry: vol spike + price < BB_lower(20, 2.5) = long
   - Exit: ATR ratio < 1.3 or RSI > 60

2. MEAN REVERSION: Connors RSI (RSI3 + RSI_Streak + PercentRank) / 3
   - CRSI < 10 = extreme oversold (long)
   - CRSI > 90 = extreme overbought (short)
   - Works best in range/bear markets (2022, 2025)

3. HTF TREND FILTER: 4h HMA(21) for bias (not hard filter)
   - Long signals stronger when price > 4h HMA
   - Short signals stronger when price < 4h HMA
   - But allow counter-trend trades during vol spikes (panic reversals)

4. REGIME DETECTION: Choppiness Index(14)
   - CHOP > 61.8 = range (favor mean reversion)
   - CHOP < 38.2 = trend (favor pullback entries)

Why this should work:
- Vol spike reversion caught the 2022 crash bottom (panic → reversal)
- Connors RSI generates MORE trades than simple RSI (3-period vs 14)
- 4h HTF provides trend context without over-filtering
- Multiple signal types = more trades across different market conditions
- 1h timeframe captures intraday moves missed by 4h/12h strategies

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_connors_4h_hma_ensemble_atr_v1"
timeframe = "1h"
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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(close, 3): Fast RSI on close
    RSI(streak, 2): RSI on consecutive up/down days
    PercentRank: Where current close ranks vs last 100 closes (0-100)
    
    CRSI < 10 = extreme oversold (long)
    CRSI > 90 = extreme overbought (short)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak
    # Streak = consecutive up/down days (+1 for up, -1 for down, 0 for flat)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values (treat streak as price)
    streak_delta = pd.Series(streak).diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Component 3: PercentRank (where current close ranks in last 100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100 * rank / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    price_range = highest_high - lowest_low
    
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.5)  # Wider bands for vol spike
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Vol spike ratio: ATR(7) / ATR(30)
    vol_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            vol_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(adx := None) or np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === REGIME DETECTION ===
        is_range_regime = chop_14[i] > 55.0  # Slightly lower threshold for more range detection
        is_trend_regime = chop_14[i] < 42.0
        
        # === HTF TREND BIAS (4h HMA) ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 2.0  # Current vol > 2x recent average
        
        # === SIGNAL 1: VOL SPIKE REVERSION ===
        # After panic (vol spike + price at BB lower), expect reversal
        vol_spike_long = vol_spike and (close[i] < bb_lower[i] * 1.01)
        vol_spike_short = vol_spike and (close[i] > bb_upper[i] * 0.99)
        
        # === SIGNAL 2: CONNORS RSI EXTREMES ===
        # CRSI < 10 = extreme oversold, CRSI > 90 = extreme overbought
        crsi_oversold = crsi[i] < 15  # Slightly relaxed for more trades
        crsi_overbought = crsi[i] > 85  # Slightly relaxed for more trades
        
        # === SIGNAL 3: RSI MEAN REVERSION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC (Multiple signal types for more trades) ===
        new_signal = 0.0
        signal_strength = 0
        
        # LONG ENTRIES
        long_score = 0
        
        # Vol spike reversal (highest priority - catches panic bottoms)
        if vol_spike_long:
            long_score += 3
            # Even stronger if CRSI also oversold
            if crsi_oversold:
                long_score += 2
        
        # Connors RSI extreme oversold
        if crsi_oversold:
            long_score += 2
            # Stronger with HTF bull bias
            if bull_bias:
                long_score += 1
        
        # Standard RSI oversold in range regime
        if rsi_oversold and is_range_regime:
            long_score += 2
            if bull_bias:
                long_score += 1
        
        # SHORT ENTRIES
        short_score = 0
        
        # Vol spike reversal at upper band
        if vol_spike_short:
            short_score += 3
            if crsi_overbought:
                short_score += 2
        
        # Connors RSI extreme overbought
        if crsi_overbought:
            short_score += 2
            if bear_bias:
                short_score += 1
        
        # Standard RSI overbought in range regime
        if rsi_overbought and is_range_regime:
            short_score += 2
            if bear_bias:
                short_score += 1
        
        # Entry threshold: need score >= 3 for entry (ensures multiple confirmations)
        if long_score >= 3:
            new_signal = SIZE
        elif short_score >= 3:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === EXIT ON SIGNAL REVERSAL ===
        # If we're long and short signal is strong, exit
        if in_position and position_side > 0 and short_score >= 4:
            stoploss_triggered = True
        if in_position and position_side < 0 and long_score >= 4:
            stoploss_triggered = True
        
        # Apply stoploss
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals