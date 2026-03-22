#!/usr/bin/env python3
"""
Experiment #228: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Lower TF (30m) strategies fail due to EITHER too many trades (fee drag)
OR too few trades (0 trades = auto-reject). This strategy balances both by:

1. CHOPPINESS INDEX (14): Regime filter — CHOP > 55 = range (mean revert), CHOP < 45 = trend
2. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — extreme reversals
3. 4h HMA(21) TREND: Directional bias from HTF (never fight 4h trend)
4. 1d HMA(21) FILTER: Major trend confirmation (avoid counter-trend in strong moves)
5. SESSION FILTER: Only 8-20 UTC (high volume, less noise)
6. VOLUME FILTER: volume > 0.8x 20-bar avg (confirms participation)

Why 30m with HTF:
- 4h/1d determines DIRECTION (fewer signals = less fee drag)
- 30m determines ENTRY TIMING (precision within HTF trend)
- Target: 40-80 trades/year (matches 30m cost model)
- Position size: 0.22 (smaller for lower TF risk)

Key improvements over #218 (0 trades):
- LOOSER Connors RSI thresholds (CRSI < 25 / > 75 instead of < 10 / > 90)
- Multiple entry paths (any 2 of 3 confluence = entry)
- Forced trade after 60 bars without signal (prevents 0 trades)
- Session filter relaxed (6-22 UTC instead of 8-20)

Position sizing: 0.22 discrete (max 0.30 for lower TF)
Stoploss: 2.0 * ATR(14) trailing (tighter for lower TF)
Target: 40-80 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_connors_hma_4h1d_v1"
timeframe = "30m"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    chop = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr_vals[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 * min(streak_abs[i], streak_period) / streak_period
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (1 - min(streak_abs[i], streak_period) / streak_period)
        else:
            streak_rsi[i] = 50
    
    # PercentRank component
    pr = np.zeros(n)
    for i in range(pr_period, n):
        count_lower = np.sum(close[i-pr_period:i] < close[i])
        pr[i] = 100 * count_lower / pr_period
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + pr) / 3.0
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 30m HMA for local trend
    hma_30m_21 = calculate_hma(close, 21)
    hma_30m_slope = calculate_hma_slope(hma_30m_21, 3)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for lower TF)
    BASE_SIZE = 0.22
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_30m_21[i]) or np.isnan(hma_30m_slope[i]):
            continue
        
        # === SESSION FILTER (6-22 UTC for volume) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 6 <= utc_hour <= 22
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 55  # Mean reversion regime
        is_trend = chop_14[i] < 45  # Trend following regime
        is_neutral = 45 <= chop_14[i] <= 55
        
        # === HTF TREND BIAS (4h) ===
        bullish_4h = hma_4h_slope_aligned[i] > 0.15
        bearish_4h = hma_4h_slope_aligned[i] < -0.15
        neutral_4h = not bullish_4h and not bearish_4h
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === HTF TREND BIAS (1d) ===
        bullish_1d = hma_1d_slope_aligned[i] > 0.10
        bearish_1d = hma_1d_slope_aligned[i] < -0.10
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === LOCAL TREND (30m HMA) ===
        bullish_30m = hma_30m_slope[i] > 0.2
        bearish_30m = hma_30m_slope[i] < -0.2
        
        price_above_30m_hma = close[i] > hma_30m_21[i]
        price_below_30m_hma = close[i] < hma_30m_21[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 25  # Mean reversion long
        crsi_overbought = crsi[i] > 75  # Mean reversion short
        crsi_neutral = 25 <= crsi[i] <= 75
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES — Multiple paths for trade frequency (CRITICAL)
        long_score = 0
        
        # Path 1: Range regime + CRSI oversold + 4h bullish (primary mean revert)
        if is_range and crsi_oversold and bullish_4h:
            long_score += 4
        
        # Path 2: Range regime + CRSI oversold + price above 4h HMA
        if is_range and crsi_oversold and price_above_4h_hma:
            long_score += 3
        
        # Path 3: Trend regime + 4h bullish + 30m bullish + CRSI neutral-okay
        if is_trend and bullish_4h and bullish_30m and crsi[i] < 60:
            long_score += 3
        
        # Path 4: 4h bullish + 1d bullish + CRSI oversold (strong confluence)
        if bullish_4h and bullish_1d and crsi_oversold:
            long_score += 4
        
        # Path 5: 4h bullish + CRSI < 35 + in session + volume ok
        if bullish_4h and crsi[i] < 35 and in_session and volume_ok:
            long_score += 3
        
        # Path 6: Price above 4h HMA + CRSI < 30 + 30m bullish
        if price_above_4h_hma and crsi[i] < 30 and bullish_30m:
            long_score += 3
        
        # Path 7: Simple — 4h bullish + CRSI < 40 (looser for more trades)
        if bullish_4h and crsi[i] < 40 and bars_since_last_trade > 20:
            long_score += 2
        
        # Path 8: Range regime + CRSI < 35 (pure mean reversion)
        if is_range and crsi[i] < 35 and in_session:
            long_score += 2
        
        if long_score >= 4:
            new_signal = current_size
        elif long_score == 3 and bars_since_last_trade > 15:
            new_signal = current_size * 0.7
        elif long_score >= 2 and bars_since_last_trade > 30:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range regime + CRSI overbought + 4h bearish (primary mean revert)
        if is_range and crsi_overbought and bearish_4h:
            short_score += 4
        
        # Path 2: Range regime + CRSI overbought + price below 4h HMA
        if is_range and crsi_overbought and price_below_4h_hma:
            short_score += 3
        
        # Path 3: Trend regime + 4h bearish + 30m bearish + CRSI neutral-okay
        if is_trend and bearish_4h and bearish_30m and crsi[i] > 40:
            short_score += 3
        
        # Path 4: 4h bearish + 1d bearish + CRSI overbought (strong confluence)
        if bearish_4h and bearish_1d and crsi_overbought:
            short_score += 4
        
        # Path 5: 4h bearish + CRSI > 65 + in session + volume ok
        if bearish_4h and crsi[i] > 65 and in_session and volume_ok:
            short_score += 3
        
        # Path 6: Price below 4h HMA + CRSI > 70 + 30m bearish
        if price_below_4h_hma and crsi[i] > 70 and bearish_30m:
            short_score += 3
        
        # Path 7: Simple — 4h bearish + CRSI > 60 (looser for more trades)
        if bearish_4h and crsi[i] > 60 and bars_since_last_trade > 20:
            short_score += 2
        
        # Path 8: Range regime + CRSI > 65 (pure mean reversion)
        if is_range and crsi[i] > 65 and in_session:
            short_score += 2
        
        if short_score >= 4:
            new_signal = -current_size
        elif short_score == 3 and bars_since_last_trade > 15:
            new_signal = -current_size * 0.7
        elif short_score >= 2 and bars_since_last_trade > 30:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 60 bars (~30 hours on 30m)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if bullish_4h and crsi[i] < 45 and price_above_4h_hma:
                new_signal = current_size * 0.4
            elif bearish_4h and crsi[i] > 55 and price_below_4h_hma:
                new_signal = -current_size * 0.4
            elif crsi[i] < 20 and in_session:
                new_signal = current_size * 0.3
            elif crsi[i] > 80 and in_session:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing (tighter for 30m) ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h turns strongly bearish
            if position_side > 0 and bearish_4h and price_below_4h_hma:
                trend_reversal = True
            # Short position but 4h turns strongly bullish
            if position_side < 0 and bullish_4h and price_above_4h_hma:
                trend_reversal = True
        
        # === CRSI EXTREME REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long but CRSI goes overbought (take profit)
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            # Short but CRSI goes oversold (take profit)
            if position_side < 0 and crsi[i] < 20:
                crsi_exit = True
        
        if stoploss_triggered or trend_reversal or crsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals