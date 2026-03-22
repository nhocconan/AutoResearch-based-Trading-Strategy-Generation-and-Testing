#!/usr/bin/env python3
"""
Experiment #313: 1d Primary + 1w HTF — HMA Trend + Connors RSI + Donchian Breakout

Hypothesis: Simpler is better for daily timeframe. Based on research showing:
1. Connors RSI (CRSI) has 75% win rate on daily - proven for ETH (Sharpe +0.923)
2. Donchian breakout + HMA trend worked for SOL (Sharpe +0.782)
3. KAMA failed in #307 (Sharpe=-0.689) - revert to HMA which is more responsive
4. Fewer conflicting conditions = more trades generated (critical lesson from 0-trade failures)
5. 1w HMA(21) for major trend, 1d HMA(8/21) for entry timing
6. ANY 2 of 3 signals can trigger entry (not all 3) - increases trade frequency
7. Aggressive frequency safeguard: force trade after 20 bars if no position

Why this might beat #307 and current best (Sharpe=0.424):
- Connors RSI is proven mean-reversion indicator (better than standard RSI)
- Donchian breakout provides clear momentum trigger
- HMA more responsive than KAMA for daily entries
- Looser entry logic (2-of-3 instead of all conditions) = more trades
- Target: 25-45 trades/year on 1d (appropriate for daily)

Position sizing: 0.30 base, 0.35 strong conviction
Stoploss: 2.5 * ATR trailing (slightly tighter than 3.0)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_connors_donchian_1w_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    More responsive than EMA with less lag.
    """
    n = period
    n2 = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=n2, min_periods=n2, adjust=False).mean()
    wma_full = close_s.ewm(span=n, min_periods=n, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile of today's price change vs last 100 days
    
    Entry: CRSI < 10 (long), CRSI > 90 (short)
    This is a proven mean-reversion indicator with ~75% win rate.
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - very short term
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    avg_loss = loss.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank (100) - percentile of today's return vs last 100 days
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    
    for i in range(period_rank, n):
        window = returns.iloc[i-period_rank:i]
        current_return = returns.iloc[i]
        if len(window) > 0:
            percent_rank[i] = 100.0 * (window < current_return).sum() / len(window)
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    for i in range(n):
        if i >= period_rank:
            crsi[i] = (rsi_3.iloc[i] + streak_rsi.iloc[i] + percent_rank[i]) / 3.0
        elif i >= period_rsi:
            crsi[i] = (rsi_3.iloc[i] + 50.0 + 50.0) / 3.0  # partial data
        else:
            crsi[i] = 50.0  # warmup
    
    return crsi

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_8 = calculate_hma(close, period=8)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.30
    LONG_STRONG = 0.35
    SHORT_BASE = 0.25
    SHORT_STRONG = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_1d_8[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        # === 1W MAJOR TREND REGIME ===
        # Bull: price above 1w HMA (favor longs)
        # Bear: price below 1w HMA (allow shorts)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        # HMA alignment
        hma_bullish = hma_1d_8[i] > hma_1d_21[i]
        hma_bearish = hma_1d_8[i] < hma_1d_21[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_1d_21[i] > hma_1d_21[i-3] if i >= 3 else False
        hma_slope_down = hma_1d_21[i] < hma_1d_21[i-3] if i >= 3 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_1d_21[i]
        price_below_hma = close[i] < hma_1d_21[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        price_below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = extremely oversold (long signal)
        # CRSI > 85 = extremely overbought (short signal)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === DONCHIAN BREAKOUT ===
        # Breakout above upper channel = momentum long
        # Breakout below lower channel = momentum short
        donchian_breakout_up = close[i] > donchian_upper[i] * 0.995
        donchian_breakout_down = close[i] < donchian_lower[i] * 1.005
        
        # === ENTRY LOGIC (ANY 2 OF 3 CONDITIONS) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Count bullish signals
            bull_signals = 0
            
            # Signal 1: CRSI oversold
            if crsi_oversold:
                bull_signals += 1
            
            # Signal 2: HMA bullish alignment
            if hma_bullish and hma_slope_up:
                bull_signals += 1
            
            # Signal 3: Donchian breakout up
            if donchian_breakout_up:
                bull_signals += 1
            
            # Signal 4: Price above HMA21 + above SMA200
            if price_above_hma and price_above_sma200:
                bull_signals += 1
            
            # Enter if 2+ bullish signals
            if bull_signals >= 2:
                if crsi_extreme_oversold or donchian_breakout_up:
                    new_signal = LONG_STRONG
                else:
                    new_signal = LONG_BASE
        
        # SHORT ENTRIES (bear regime)
        if regime_bear:
            # Count bearish signals
            bear_signals = 0
            
            # Signal 1: CRSI overbought
            if crsi_overbought:
                bear_signals += 1
            
            # Signal 2: HMA bearish alignment
            if hma_bearish and hma_slope_down:
                bear_signals += 1
            
            # Signal 3: Donchian breakout down
            if donchian_breakout_down:
                bear_signals += 1
            
            # Signal 4: Price below HMA21 + below SMA200
            if price_below_hma and price_below_sma200:
                bear_signals += 1
            
            # Enter if 2+ bearish signals
            if bear_signals >= 2:
                if crsi_extreme_overbought or donchian_breakout_down:
                    new_signal = -SHORT_STRONG
                else:
                    new_signal = -SHORT_BASE
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 1d) ===
        # Force trade if no signal for 20 bars (~20 days)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 40.0:
                new_signal = LONG_BASE * 0.7
            elif regime_bear and crsi[i] > 60.0:
                new_signal = -SHORT_BASE * 0.7
            elif crsi_extreme_oversold:
                new_signal = LONG_BASE * 0.7
            elif crsi_extreme_overbought:
                new_signal = -SHORT_BASE * 0.7
            elif hma_bullish and price_above_hma:
                new_signal = LONG_BASE * 0.7
            elif hma_bearish and price_below_hma:
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
            # Long position: exit when HMA turns bearish + price below
            if position_side > 0 and hma_bearish and price_below_hma:
                hma_exit = True
            # Short position: exit when HMA turns bullish + price above
            if position_side < 0 and hma_bullish and price_above_hma:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1w regime turns bearish + price below HMA
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1w regime turns bullish + price above HMA
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.15:
                new_signal = 0.0
            elif new_signal > 0.32:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.27:
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