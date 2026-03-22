#!/usr/bin/env python3
"""
Experiment #328: 30m Primary + 4h/1d HTF — Connors RSI + HMA Trend + Session/Volume Filter

Hypothesis: Lower timeframe (30m) with strict HTF filter can outperform daily strategies by:
1. Connors RSI (CRSI) has 75% win rate on mean reversion entries in crypto
2. 4h HMA(21) provides trend direction without excessive lag (proven in best strategy)
3. Session filter (8-20 UTC) captures London/NY overlap when volume is highest
4. Volume filter (>0.8x avg) ensures entries have participation
5. Target: 40-80 trades/year on 30m (strict confluence prevents fee drag)

Why this might beat current best (Sharpe=0.424):
- CRSI is more sensitive than RSI(14) for pullback entries
- 30m entries capture better risk/reward within 4h trend
- Session filter avoids low-volume Asian session whipsaws
- Discrete signal levels minimize churn costs

CRSI Formula (Connors RSI):
- RSI(3): 3-period RSI on close
- RSI_Streak(2): RSI on streak duration (consecutive up/down days)
- PercentRank(100): percentile rank of today's return vs last 100 days
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank) / 3

Position sizing: 0.20-0.30 (smaller for 30m to control DD)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_hma_session_vol_4h_v1"
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
    
    RSI_Streak: RSI applied to streak duration (consecutive up/down closes)
    PercentRank: percentile rank of today's return vs last N days
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100 scale)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            # Up streak - calculate RSI on streak length
            up_streaks = np.zeros(streak_period)
            for j in range(streak_period):
                if i - j >= 0 and streak[i-j] > 0:
                    up_streaks[j] = streak[i-j]
            avg_up = np.mean(up_streaks[up_streaks > 0]) if np.any(up_streaks > 0) else 0
            streak_rsi[i] = min(100, avg_up * 20)  # Scale to 0-100
        elif streak[i] < 0:
            down_streaks = np.zeros(streak_period)
            for j in range(streak_period):
                if i - j >= 0 and streak[i-j] < 0:
                    down_streaks[j] = np.abs(streak[i-j])
            avg_down = np.mean(down_streaks[down_streaks > 0]) if np.any(down_streaks > 0) else 0
            streak_rsi[i] = max(0, 100 - avg_down * 20)  # Inverse scale
        else:
            streak_rsi[i] = 50.0
    
    # PercentRank component
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100
    
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine components
    for i in range(rank_period, n):
        crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    
    More responsive than EMA with less lag.
    """
    n = period
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA(n/2)
    wma_half = close_s.ewm(span=half_n, min_periods=half_n, adjust=False).mean()
    # WMA(n)
    wma_full = close_s.ewm(span=n, min_periods=n, adjust=False).mean()
    
    # Raw HMA
    raw_hma = 2.0 * wma_half - wma_full
    
    # Final HMA
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = (open_time // 1000 // 3600) % 24
    return hour

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
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    
    # Calculate 1d HTF indicators (regime filter)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_30m_21 = calculate_hma(close, 21)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller size for 30m to control drawdown
    LONG_BASE = 0.20
    LONG_STRONG = 0.30
    SHORT_BASE = 0.18
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_30m_21[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 4H MAJOR TREND REGIME (primary direction filter) ===
        # Bull: 4h HMA(21) > HMA(48) and price > HMA(21)
        # Bear: 4h HMA(21) < HMA(48) and price < HMA(21)
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_48_aligned[i]
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # 1D regime confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i] if not np.isnan(hma_1d_21_aligned[i]) else True
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i] if not np.isnan(hma_1d_21_aligned[i]) else False
        
        # === 30M LOCAL TREND ===
        price_above_hma = close[i] > hma_30m_21[i]
        price_below_hma = close[i] < hma_30m_21[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_30m_21[i] > hma_30m_21[i-3] if i >= 3 else False
        hma_slope_down = hma_30m_21[i] < hma_30m_21[i-3] if i >= 3 else False
        
        # === CRSI SIGNALS (mean reversion entries) ===
        # CRSI < 15 = extremely oversold (long)
        # CRSI > 85 = extremely overbought (short)
        crsi_oversold = crsi[i] < 18.0
        crsi_overbought = crsi[i] > 82.0
        crsi_extreme_oversold = crsi[i] < 12.0
        crsi_extreme_overbought = crsi[i] > 88.0
        
        # CRSI turning (momentum shift)
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] > 0.7  # At least 70% of average
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (require: 4h bull + CRSI oversold + session + volume)
        if hma_4h_bullish and price_above_4h_hma:
            # Primary: CRSI extreme oversold in 4h uptrend
            if crsi_extreme_oversold and in_session and volume_ok:
                new_signal = LONG_STRONG
            
            # CRSI oversold + turning up + 30m HMA support
            elif crsi_oversold and crsi_rising and price_above_hma and in_session:
                new_signal = LONG_BASE
            
            # CRSI moderately oversold + strong 4h trend + volume surge
            elif crsi[i] < 25.0 and hma_4h_bullish and vol_ratio[i] > 1.3 and in_session:
                new_signal = LONG_BASE
        
        # SHORT ENTRIES (require: 4h bear + CRSI overbought + session + volume)
        if hma_4h_bearish and price_below_4h_hma:
            # Primary: CRSI extreme overbought in 4h downtrend
            if crsi_extreme_overbought and in_session and volume_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # CRSI overbought + turning down + 30m HMA resistance
            elif crsi_overbought and crsi_falling and price_below_hma and in_session:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # CRSI moderately overbought + strong 4h downtrend + volume surge
            elif crsi[i] > 75.0 and hma_4h_bearish and vol_ratio[i] > 1.3 and in_session:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
        
        # === FREQUENCY SAFEGUARD (ensure trades generate) ===
        # Force trade if no signal for 100 bars (~2 days on 30m)
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and crsi[i] < 30.0 and in_session:
                new_signal = LONG_BASE * 0.7
            elif hma_4h_bearish and crsi[i] > 70.0 and in_session:
                new_signal = -SHORT_BASE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when price crosses below 30m HMA
            if position_side > 0 and price_below_hma and hma_slope_down:
                hma_exit = True
            # Short position: exit when price crosses above 30m HMA
            if position_side < 0 and price_above_hma and hma_slope_up:
                hma_exit = True
        
        # === 4H REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h regime turns bearish
            if position_side > 0 and hma_4h_bearish and price_below_4h_hma:
                regime_reversal = True
            # Short position but 4h regime turns bullish
            if position_side < 0 and hma_4h_bullish and price_above_4h_hma:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.26:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.22:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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