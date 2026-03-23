#!/usr/bin/env python3
"""
Experiment #400: 1h Primary + 4h/12h HTF — ADX Regime + CRSI + HMA Trend

Hypothesis: Using ADX for regime detection (simpler than CHOP which caused 0 trades)
combined with Connors RSI for entries and multi-HMA trend structure will generate
30-60 trades/year with Sharpe > 0.612.

Key innovations vs failed 1h strategies (#390, #395, #398 all got Sharpe=0.000):
1. ADX(14) regime: >25 = trend, <20 = range (hysteresis 18-25 to avoid flip-flop)
2. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Proven 75% win rate in bear/range markets (ETH Sharpe +0.923 in notes)
3. 12h HMA(21) for overall bias (more stable than 1d, less lag)
4. 4h HMA(16/48) for intermediate trend confirmation
5. 1h CRSI for entry timing (extreme <15 long, >85 short)
6. NO session filter (caused 0 trades in #390, #395, #398)
7. Relaxed entry: only need HTF bias + CRSI extreme + ADX regime agreement
8. Asymmetric sizing: 0.25 in trend, 0.20 in range (lower risk in chop)

Why this should beat Sharpe=0.612:
- CRSI proven edge in bear markets (2022 crash, 2025 range)
- ADX simpler than CHOP = more reliable regime detection
- 1h TF with HTF direction = 30-60 trades/year target (not 0, not 200+)
- Different signal combo than #399 (which got Sharpe=-0.019)
- Fewer filters than failed 1h strategies = actual trades generated

Target: Sharpe > 0.612, 30-60 trades/year, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_adx_crsi_hma_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using Wilder's smoothing."""
    n = len(close)
    adx = np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive up/down days (positive for up, negative for down)
    PercentRank: percentile of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (map to 0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        gains = np.sum(streak_window[streak_window > 0])
        losses = np.abs(np.sum(streak_window[streak_window < 0]))
        if losses < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = gains / (losses + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10)
    
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1)
        percent_rank[i] = rank * 100.0
    
    # Combine
    for i in range(rank_period, n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, period=14)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14_1h = calculate_atr(high, low, close, period=14)
    
    # Calculate 4h HMA trend
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, 48)
    hma_4h_16 = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48 = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    # Calculate 12h HMA for overall bias
    hma_12h_21_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_21 = align_htf_to_ltf(prices, df_12h, hma_12h_21_raw)
    
    signals = np.zeros(n)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(adx_1h[i]) or np.isnan(crsi_1h[i]) or np.isnan(atr_14_1h[i]):
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]) or np.isnan(hma_12h_21[i]):
            continue
        if atr_14_1h[i] <= 1e-10:
            continue
        
        # === ADX REGIME (with hysteresis) ===
        # ADX > 25 = trending, ADX < 20 = ranging
        # Use hysteresis to avoid flip-flop
        is_trending = adx_1h[i] > 25.0
        is_ranging = adx_1h[i] < 20.0
        # Neutral: 20 <= ADX <= 25 (hold current bias)
        
        # === HTF BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_21[i]
        price_below_hma_12h = close[i] < hma_12h_21[i]
        
        # === INTERMEDIATE TREND (4h HMA crossover) ===
        hma_4h_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_4h_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === CRSI EXTREMES ===
        # CRSI < 15 = extremely oversold (long opportunity)
        # CRSI > 85 = extremely overbought (short opportunity)
        crsi_oversold = crsi_1h[i] < 15.0
        crsi_overbought = crsi_1h[i] > 85.0
        
        # === POSITION SIZING ===
        # Smaller size in range market (more risk of whipsaw)
        if is_trending:
            base_size = 0.25  # 25% in trending regime
        else:
            base_size = 0.20  # 20% in ranging/neutral regime
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        # Need: 12h bullish OR 4h bullish + CRSI oversold
        long_bias = price_above_hma_12h or (hma_4h_bullish and price_above_hma_12h == False)
        
        if long_bias and crsi_oversold:
            # Strong long: HTF bias + CRSI extreme
            desired_signal = base_size
        elif hma_4h_bullish and crsi_oversold and is_ranging:
            # Mean reversion long in range with 4h support
            desired_signal = base_size * 0.8
        
        # SHORT SETUP
        short_bias = price_below_hma_12h or (hma_4h_bearish and price_below_hma_12h == False)
        
        if short_bias and crsi_overbought:
            # Strong short: HTF bias + CRSI extreme
            desired_signal = -base_size
        elif hma_4h_bearish and crsi_overbought and is_ranging:
            # Mean reversion short in range with 4h resistance
            desired_signal = -base_size * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXIT (extreme reversal) ===
        if in_position and position_side > 0 and crsi_1h[i] > 80.0:
            # Long exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_1h[i] < 20.0:
            # Short exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === HTF BIAS EXIT ===
        if in_position and position_side > 0 and price_below_hma_12h and hma_4h_bearish:
            # Both HTF and intermediate turned bearish
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_12h and hma_4h_bullish:
            # Both HTF and intermediate turned bullish
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and long_bias:
                desired_signal = base_size
            elif position_side < 0 and short_bias:
                desired_signal = -base_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals