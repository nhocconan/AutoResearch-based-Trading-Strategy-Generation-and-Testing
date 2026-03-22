#!/usr/bin/env python3
"""
Experiment #008: 30m Connors RSI + 4h/1d HMA Trend + Session/Volume Filter

Hypothesis: 30m primary with strict confluence filters will generate 30-80 trades/year
while maintaining positive Sharpe across BTC/ETH/SOL. Key design:
1. 1d HMA(48) for major regime (bull/bear) - call ONCE before loop
2. 4h HMA(21) for intermediate trend direction - call ONCE before loop
3. Choppiness Index(14) on 30m for regime detection (range vs trend)
4. Connors RSI(3,2,100) for entry timing (more responsive than standard RSI)
5. Session filter: only 8-20 UTC (high liquidity, avoid Asian chop)
6. Volume filter: volume > 0.8x 20-period average
7. ATR(14) stoploss at 2.5x for risk management
8. Discrete sizing: 0.20 base, 0.25 strong confluence

Why this should work:
- Connors RSI has 75% win rate in backtests (Larry Connors research)
- 4h/1d HMA filters prevent counter-trend trades (major failure mode)
- Session filter avoids low-liquidity periods (reduces false signals)
- Volume confirmation ensures real moves, not noise
- 30m TF with strict filters targets 30-80 trades/year (optimal fee efficiency)
- Smaller position size (0.20-0.25) appropriate for lower TF volatility

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.25 discrete (smaller for lower TF)
Stoploss: 2.5 * ATR(14)
Trade frequency target: 30-80 trades/year (CRITICAL for 30m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_connors_4h_1d_hma_session_vol_v1"
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
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) on close prices
    2. RSI(2) on streak (consecutive up/down days)
    3. PercentRank(100) of price change over 100 periods
    
    Entry signals:
    - Long: CRSI < 10 (oversold)
    - Short: CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to binary for RSI calculation
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_avg = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_avg = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_loss_avg = np.where(streak_loss_avg == 0, 1e-10, streak_loss_avg)
    streak_rsi = 100 - (100 / (1 + streak_gain_avg / streak_loss_avg))
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: PercentRank(100) of price change
    price_change = np.diff(close)
    price_change = np.insert(price_change, 0, 0)
    
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = price_change[i-rank_period+1:i+1]
        current = price_change[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / rank_period * 100
    
    # Combine components
    for i in range(max(rsi_period, streak_period, rank_period), n):
        crsi[i] = (rsi_close[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    # Convert to hours UTC
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
    
    # Calculate 1d HMA trend (regime filter)
    hma_1d_48 = calculate_hma(df_1d['close'].values, 48)
    hma_1d_48_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_48)
    
    # Calculate 4h HMA trend (direction filter)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    vol_avg_20 = calculate_volume_avg(volume, 20)
    
    # Also calculate 30m HMA for local trend
    hma_30m_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, smaller for 30m)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100  # Ensure first trades can happen
    
    for i in range(150, n):  # Need 150 bars for Connors RSI rank_period
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_48_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === UTC SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 1D HTF REGIME (bull/bear) ===
        regime_bullish = close[i] > hma_1d_48_aligned[i]
        regime_bearish = close[i] < hma_1d_48_aligned[i]
        
        # === 4H HTF TREND (direction) ===
        htf_bullish = close[i] > hma_4h_21_aligned[i]
        htf_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 30M LOCAL TREND ===
        local_bullish = close[i] > hma_30m_21[i]
        local_bearish = close[i] < hma_30m_21[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 55  # Range market (mean revert)
        chop_trend = chop_14[i] < 45  # Trending market (trend follow)
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15  # Long entry
        crsi_overbought = crsi[i] > 85  # Short entry
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # === POSITION SIZING BASED ON CONFLUENCE ===
        # Count confluence factors (HTF trend + local trend + regime alignment)
        confluence_long = sum([htf_bullish, local_bullish, regime_bullish])
        confluence_short = sum([htf_bearish, local_bearish, regime_bearish])
        
        current_size = BASE_SIZE
        if confluence_long >= 3 or confluence_short >= 3:
            current_size = STRONG_SIZE
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Multiple confluence scenarios
        # Scenario 1: Trending market + HTF bullish + CRSI oversold + session + volume
        if chop_trend and htf_bullish and crsi_oversold and in_session and volume_confirmed:
            new_signal = current_size
        
        # Scenario 2: Range market + HTF bullish + CRSI extreme oversold + session
        elif chop_range and regime_bullish and crsi_extreme_oversold and in_session:
            new_signal = current_size
        
        # Scenario 3: All HTF aligned + CRSI oversold (strongest signal)
        elif regime_bullish and htf_bullish and local_bullish and crsi_oversold and in_session:
            new_signal = STRONG_SIZE
        
        # SHORT ENTRY: Multiple confluence scenarios
        # Scenario 1: Trending market + HTF bearish + CRSI overbought + session + volume
        if chop_trend and htf_bearish and crsi_overbought and in_session and volume_confirmed:
            new_signal = -current_size
        
        # Scenario 2: Range market + HTF bearish + CRSI extreme overbought + session
        elif chop_range and regime_bearish and crsi_extreme_overbought and in_session:
            new_signal = -current_size
        
        # Scenario 3: All HTF aligned + CRSI overbought (strongest signal)
        elif regime_bearish and htf_bearish and local_bearish and crsi_overbought and in_session:
            new_signal = -STRONG_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~25 hours on 30m), allow weaker entry
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if regime_bullish and htf_bullish and crsi_oversold and in_session:
                new_signal = BASE_SIZE * 0.8
            elif regime_bearish and htf_bearish and crsi_overbought and in_session:
                new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === CRSI MEAN REVERSION EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought (mean reversion)
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            # Exit short when CRSI becomes oversold (mean reversion)
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit long if 1d regime turns bearish
            if position_side > 0 and regime_bearish:
                regime_exit = True
            # Exit short if 1d regime turns bullish
            if position_side < 0 and regime_bullish:
                regime_exit = True
        
        # === SESSION EXIT ===
        # Close positions outside session hours to avoid overnight risk
        session_exit = False
        if in_position and not in_session:
            session_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or crsi_exit or regime_exit or session_exit:
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