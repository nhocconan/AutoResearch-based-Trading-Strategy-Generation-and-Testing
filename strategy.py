#!/usr/bin/env python3
"""
Experiment #037: 1d Primary + 4h HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Daily timeframe with 4h trend bias and Connors RSI entries will beat the current
best (Sharpe=0.486). Key improvements over #027:
1. Connors RSI (CRSI) instead of regular RSI - proven 75% win rate in mean reversion
2. 4h HMA instead of 1w HMA - more responsive to trend changes while still filtering noise
3. Asymmetric sizing: 0.35 in high-confidence setups, 0.25 in moderate setups
4. Looser entry thresholds to guarantee trade generation (CRSI<15 instead of RSI<30)
5. Volume confirmation on breakouts to reduce false signals

Strategy Logic:
1. CHOPPINESS INDEX (14): CHOP > 55 = range regime (mean revert), CHOP < 45 = trend regime
2. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long entry: CRSI < 15 (oversold extreme)
   - Short entry: CRSI > 85 (overbought extreme)
3. 4h HMA(21): Macro trend bias - only take longs when price > 4h HMA, shorts when < 4h HMA
4. TREND REGIME: Donchian(20) breakout + 4h HMA confirmation + volume spike
5. RANGE REGIME: CRSI extremes + Bollinger Band touches + 4h HMA helps
6. ATR(14) trailing stoploss: 2.5*ATR to protect capital

Why this should beat current best:
- CRSI is more sensitive than RSI for short-term extremes (proven in literature)
- 4h HTF is more responsive than 1w for crypto volatility
- Asymmetric sizing reduces risk on marginal setups
- Volume filter on breakouts reduces false signals in low-liquidity periods

Position size: 0.25-0.35 (discrete, within 0.20-0.40 range)
Stoploss: 2.5*ATR trailing
Target trades: 25-45/year on 1d timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_regime_4h_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_streak_rsi(close, period=2):
    """
    Calculate RSI of consecutive up/down days (Connors RSI component).
    Streak = number of consecutive days price moved in same direction.
    """
    n = len(close)
    streak = np.zeros(n)
    direction = np.zeros(n)  # 1 = up, -1 = down, 0 = neutral
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            direction[i] = 1
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            direction[i] = -1
            if direction[i-1] == -1:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            direction[i] = 0
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    abs_streak = np.abs(streak)
    streak_rsi = np.zeros(n)
    
    for i in range(period, n):
        if streak[i] >= 0:
            # Bullish streak - higher is more overbought
            streak_rsi[i] = min(100.0, streak[i] * 25.0)
        else:
            # Bearish streak - more negative is more oversold
            streak_rsi[i] = max(0.0, 100.0 + streak[i] * 25.0)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank (Connors RSI component).
    Percent of days in lookback where close was lower than current close.
    """
    n = len(close)
    percent_rank = np.zeros(n)
    
    for i in range(period, n):
        lookback = close[i-period+1:i+1]
        count_lower = np.sum(lookback[:-1] < close[i])
        percent_rank[i] = (count_lower / (period - 1)) * 100.0
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_streak_rsi(close, period=streak_period)
    percent_rank = calculate_percent_rank(close, period=pr_period)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes (volume > 1.5x 20-day average)."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    volume_spike = volume > (vol_sma * 1.5)
    return volume_spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    volume_spike = calculate_volume_spike(volume, period=20)
    
    # Calculate HMA for trend confirmation
    hma_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_HIGH = 0.35  # High confidence setups
    POSITION_SIZE_MED = 0.25   # Moderate confidence setups
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(120, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(hma_21[i]) or atr_14[i] == 0:
            continue
        
        # === 4H MACRO BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market
        is_trending = chop_value < 45.0  # Trend market (with hysteresis)
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === HMA TREND ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        hma_slope_up = hma_21[i] > hma_21[i-5] if i > 5 else False
        hma_slope_down = hma_21[i] < hma_21[i-5] if i > 5 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        confidence = "medium"  # Default confidence level
        
        # --- RANGING REGIME: Mean Reversion (CRSI extremes) ---
        if is_ranging:
            # Long: CRSI oversold extreme + price near BB lower + 4h helps
            if crsi_oversold or (price_below_bb_lower and crsi[i] < 25):
                if price_above_hma_4h or crsi_rising:  # 4h bullish OR CRSI turning up
                    new_signal = POSITION_SIZE_HIGH if crsi_oversold else POSITION_SIZE_MED
                    confidence = "high" if crsi_oversold else "medium"
            
            # Short: CRSI overbought extreme + price near BB upper + 4h helps
            elif crsi_overbought or (price_above_bb_upper and crsi[i] > 75):
                if price_below_hma_4h or crsi_falling:  # 4h bearish OR CRSI turning down
                    new_signal = -POSITION_SIZE_HIGH if crsi_overbought else -POSITION_SIZE_MED
                    confidence = "high" if crsi_overbought else "medium"
        
        # --- TRENDING REGIME: Trend Following (Donchian breakout) ---
        elif is_trending:
            # Long: Donchian breakout + HMA bullish + 4h confirms + volume spike
            if donchian_breakout_long and hma_bullish:
                if price_above_hma_4h and hma_slope_up:  # 4h + daily trend aligned
                    if volume_spike[i]:  # Volume confirmation
                        new_signal = POSITION_SIZE_HIGH
                        confidence = "high"
                    else:
                        new_signal = POSITION_SIZE_MED
                        confidence = "medium"
            
            # Short: Donchian breakdown + HMA bearish + 4h confirms + volume spike
            elif donchian_breakout_short and hma_bearish:
                if price_below_hma_4h and hma_slope_down:  # 4h + daily trend aligned
                    if volume_spike[i]:  # Volume confirmation
                        new_signal = -POSITION_SIZE_HIGH
                        confidence = "high"
                    else:
                        new_signal = -POSITION_SIZE_MED
                        confidence = "medium"
        
        # --- NEUTRAL REGIME: HMA crossover with CRSI filter ---
        if new_signal == 0.0 and not is_ranging and not is_trending:
            # Long: Price crosses above HMA + CRSI rising + 4h helps
            if close[i] > hma_21[i] and close[i-1] <= hma_21[i-1]:
                if crsi_rising and (price_above_hma_4h or crsi[i] < 50):
                    new_signal = POSITION_SIZE_MED
            
            # Short: Price crosses below HMA + CRSI falling + 4h helps
            elif close[i] < hma_21[i] and close[i-1] >= hma_21[i-1]:
                if crsi_falling and (price_below_hma_4h or crsi[i] > 50):
                    new_signal = -POSITION_SIZE_MED
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME/TREND CHANGE ===
        # Exit long if 4h trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_bearish and chop_value < 40:
                new_signal = 0.0
        
        # Exit short if 4h trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_bullish and chop_value < 40:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals