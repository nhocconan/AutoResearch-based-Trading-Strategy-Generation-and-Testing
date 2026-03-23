#!/usr/bin/env python3
"""
Experiment #700: 1h Primary + 4h/12h HTF — Connors RSI Mean Reversion + Vol-Adjusted BB

Hypothesis: After 608 failed strategies, the pattern shows:
1. Fisher Transform strategies (#695, #696) both failed — reversal indicators alone don't work
2. Complex multi-filter strategies (#688-#692, #698-#699) all failed — overfiltering kills trades
3. #693 (1d CHOP+CRSI+HMA+1w) worked best — simpler regime + mean-reversion is the key
4. 1h strategies fail because they generate TOO MANY trades → fee drag

This strategy uses CONNORS RSI (CRSI) which has 75% win rate in literature:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 15 + price > SMA(200) + 4h HMA bullish
- Short: CRSI > 85 + price < SMA(200) + 4h HMA bearish

KEY DIFFERENCES from failed attempts:
- Fewer filters: only CRSI + HTF trend + BB confirmation (3 confluence, not 5+)
- Looser CRSI thresholds: <15/>85 (not <10/>90) to ensure trades generate
- Vol-adjusted BB: wider bands in high vol (2.5 std), tighter in low vol (1.8 std)
- Position size: 0.20 (conservative for 1h to reduce fee impact)
- Target: 40-60 trades/year (strict enough to avoid fee drag, loose enough to trade)

Why this might beat Sharpe=0.520:
- CRSI is proven mean-reversion indicator (Larry Connors research)
- 4h HMA provides trend bias without overfiltering
- Vol-adjusted BB adapts to market conditions (wider in panic, tighter in calm)
- Asymmetric: only short when 4h bearish (avoids shorting bull rallies)

Position sizing: 0.20 discrete (conservative for 1h per Rule 4 & trade frequency limits)
Stoploss: 2.5*ATR trailing
Target trade frequency: 40-60/year on 1h (within 30-60 limit per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_volbb_hma4h_regime_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of up/down streak lengths
    3. PercentRank(100): Percentile rank of price change over 100 bars
    
    Signals (from Connors research):
    - Long: CRSI < 10-15 (oversold)
    - Short: CRSI > 85-90 (overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streaks
    # Calculate streak lengths (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    streak[0] = 0
    
    for i in range(1, n):
        if delta.iloc[i] > 0:
            if delta.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta.iloc[i] < 0:
            if delta.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert streaks to positive values for RSI calculation
    streak_positive = np.abs(streak)
    streak_positive[streak_positive == 0] = 1  # Avoid zeros
    
    # Calculate RSI on streak lengths (use 2-period as per Connors)
    # We need to treat streak direction separately
    streak_up = np.where(streak > 0, streak, 0)
    streak_down = np.where(streak < 0, -streak, 0)
    
    # Simplified: use rolling mean of streak magnitude
    streak_sma = pd.Series(streak_positive).rolling(window=streak_period, min_periods=streak_period).mean()
    
    # Create pseudo-RSI for streaks (higher streak = more extreme)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            # Up streak - calculate how extreme
            avg_up = streak_up[max(0,i-streak_period):i+1].mean()
            avg_down = streak_down[max(0,i-streak_period):i+1].mean()
            if avg_down == 0:
                streak_rsi[i] = 100.0
            else:
                rs = avg_up / avg_down
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            # Down streak
            avg_down = streak_down[max(0,i-streak_period):i+1].mean()
            avg_up = streak_up[max(0,i-streak_period):i+1].mean()
            if avg_up == 0:
                streak_rsi[i] = 0.0
            else:
                rs = avg_up / avg_down
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank of price changes
    pct_change = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = pct_change.iloc[i-rank_period:i]
        current = pct_change.iloc[i]
        if np.isnan(current):
            percent_rank[i] = 50.0
        else:
            rank = np.sum(window < current)
            percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def calculate_vol_adjusted_bb(close, period=20, atr_period=14):
    """
    Calculate Volatility-Adjusted Bollinger Bands.
    
    In high volatility (ATR > 1.5x avg), use wider bands (2.5 std)
    In low volatility (ATR < 0.8x avg), use tighter bands (1.8 std)
    """
    close_s = pd.Series(close)
    
    # Calculate ATR for vol regime
    # Simplified: use rolling std as proxy for ATR
    rolling_std = close_s.rolling(window=atr_period, min_periods=atr_period).std()
    avg_std = rolling_std.rolling(window=50, min_periods=50).mean()
    
    # Vol regime: high if current std > 1.3x average
    vol_ratio = rolling_std / (avg_std + 1e-10)
    
    # Adjust std_dev based on volatility
    std_dev = np.where(vol_ratio > 1.3, 2.5, 
              np.where(vol_ratio < 0.8, 1.8, 2.0))
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    # Apply variable std_dev
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # %B (position within bands)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    return upper, lower, pct_b

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, period=200)
    bb_upper, bb_lower, bb_pct_b = calculate_vol_adjusted_bb(close, period=20, atr_period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 1h)
    POSITION_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # ADX-like trend strength (simplified using HMA slope)
    prev_trend_strength = 0
    
    for i in range(250, n):  # Start after warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(bb_pct_b[i]) or np.isnan(sma_200[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS ===
        # HMA slope over last 10 bars (4h = slower trend signal)
        hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-10] if i >= 10 else 0
        hma_4h_bullish = hma_4h_slope > 0
        hma_4h_bearish = hma_4h_slope < 0
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # Price relative to 200 SMA (long-term trend filter)
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CRSI SIGNALS (Connors RSI mean reversion) ===
        # Looser thresholds to ensure trades generate (target 40-60/year)
        crsi_oversold = crsi[i] < 15.0  # Long signal
        crsi_overbought = crsi[i] > 85.0  # Short signal
        
        # === BOLLINGER BAND CONFIRMATION ===
        bb_touch_lower = bb_pct_b[i] < 0.10  # At or below lower band
        bb_touch_upper = bb_pct_b[i] > 0.90  # At or above upper band
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Mean reversion: CRSI oversold + BB lower + 4h not strongly bearish
        if crsi_oversold and bb_touch_lower:
            # Only long if 4h trend is neutral or bullish (avoid catching falling knife)
            if hma_4h_bullish or (not hma_4h_bearish and price_above_sma200):
                new_signal = POSITION_SIZE
        
        # Trend pullback: 4h bullish + CRSI moderate oversold + price > SMA200
        elif hma_4h_bullish and price_above_hma_4h and price_above_sma200:
            if crsi[i] < 35.0 and bb_pct_b[i] < 0.35:  # Pullback within uptrend
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Mean reversion: CRSI overbought + BB upper + 4h not strongly bullish
        if crsi_overbought and bb_touch_upper:
            # Only short if 4h trend is neutral or bearish
            if hma_4h_bearish or (not hma_4h_bullish and price_below_sma200):
                new_signal = -POSITION_SIZE
        
        # Trend pullback: 4h bearish + CRSI moderate overbought + price < SMA200
        elif hma_4h_bearish and price_below_hma_4h and price_below_sma200:
            if crsi[i] > 65.0 and bb_pct_b[i] > 0.65:  # Rally within downtrend
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If we're in a position and no new signal, hold (don't flip to 0)
        if in_position and new_signal == 0.0:
            # Keep existing position unless stoploss hits
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long position: trail stop below highest high
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short position: trail stop above lowest low
            if lowest_since_entry == 0.0:
                lowest_since_entry = low[i]
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        # Exit long if 4h turns strongly bearish
        if in_position and position_side > 0:
            if hma_4h_bearish and price_below_hma_4h and hma_4h_slope < -0.02 * hma_4h_aligned[i]:
                new_signal = 0.0
        
        # Exit short if 4h turns strongly bullish
        if in_position and position_side < 0:
            if hma_4h_bullish and price_above_hma_4h and hma_4h_slope > 0.02 * hma_4h_aligned[i]:
                new_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit on mean reversion) ===
        # Exit long when CRSI goes overbought (mean reversion complete)
        if in_position and position_side > 0:
            if crsi[i] > 75.0:
                new_signal = 0.0
        
        # Exit short when CRSI goes oversold (mean reversion complete)
        if in_position and position_side < 0:
            if crsi[i] < 25.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else np.inf
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else np.inf
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals