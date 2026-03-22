#!/usr/bin/env python3
"""
Experiment #193: 15m Trend Pullback + 4h HMA Bias + 1h RSI Momentum + ATR Stop

Hypothesis: 15m timeframe captures intraday pullbacks within the 4h trend direction.
By using 4h HMA for stable trend bias, 1h RSI for momentum confirmation, and 15m
EMA for entry timing, we can catch more frequent trades than 4h/12h strategies
while maintaining directional edge. This should generate 20-50 trades/year with
better risk/reward than pure breakout strategies.

Why 15m might work where 4h/12h failed:
- 15m = 96 bars/day, captures intraday mean reversion within trends
- 4h HMA (21) = stable trend filter, avoids counter-trend trades
- 1h RSI pullback = enters on weakness in uptrend, strength in downtrend
- More trade opportunities = better statistical significance
- ATR stoploss (2.0x) protects against 15m whipsaws

Learning from failures:
- #187 (15m Supertrend): Sharpe=-1.239 - Supertrend alone whipsaws
- #181 (15m CRSI): Sharpe=-5.141 - mean reversion without trend filter fails
- #190 (4h Fisher): 0 trades - conditions too strict
- Key insight: Need HTF trend filter + LTF pullback entry (not pure breakout)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA trend bias + 1h RSI momentum (both via mtf_data helper)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pullback_4h_hma_1h_rsi_atr_v1"
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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average for trend filter."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    rsi_15m = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    profit_target_hit = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # SMA200 filter for long-term trend
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === 1H RSI MOMENTUM ===
        # RSI 1h oversold in uptrend = long opportunity
        # RSI 1h overbought in downtrend = short opportunity
        rsi_1h_oversold = rsi_1h_aligned[i] < 45
        rsi_1h_overbought = rsi_1h_aligned[i] > 55
        
        # === 15M RSI PULLBACK ===
        # More extreme levels for entry trigger
        rsi_15m_oversold = rsi_15m[i] < 40
        rsi_15m_overbought = rsi_15m[i] > 60
        
        # === EMA STRUCTURE ===
        # EMA21 > EMA50 = bullish structure
        # EMA21 < EMA50 = bearish structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === PRICE VS EMA21 ===
        # Pullback to EMA21 in uptrend
        price_near_ema21_long = close[i] <= ema_21[i] * 1.005  # Within 0.5% of EMA21
        price_near_ema21_short = close[i] >= ema_21[i] * 0.995  # Within 0.5% of EMA21
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + SMA200 filter + 1h RSI not overbought + 15m RSI oversold + EMA structure
        # Relaxed conditions to ensure enough trades
        if bull_trend_4h and above_sma200:
            if rsi_1h_oversold and rsi_15m_oversold:
                if ema_bullish or price_near_ema21_long:
                    new_signal = SIZE_BASE
        
        # Short: 4h bearish + SMA200 filter + 1h RSI not oversold + 15m RSI overbought + EMA structure
        if bear_trend_4h and below_sma200:
            if rsi_1h_overbought and rsi_15m_overbought:
                if ema_bearish or price_near_ema21_short:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
                # Take profit at 2R: reduce to half position
                if not profit_target_hit:
                    profit_target = entry_price + 4.0 * atr[int(i)]  # 2R = 2 * 2ATR
                    if close[i] >= profit_target:
                        profit_target_hit = True
                        new_signal = SIZE_HALF  # Reduce to half position
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
                # Take profit at 2R: reduce to half position
                if not profit_target_hit:
                    profit_target = entry_price - 4.0 * atr[int(i)]  # 2R = 2 * 2ATR
                    if close[i] <= profit_target:
                        profit_target_hit = True
                        new_signal = -SIZE_HALF  # Reduce to half position
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and bull_trend_4h:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                profit_target_hit = False
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                profit_target_hit = False
            elif abs(new_signal) < abs(signals[i-1]) if i > 0 else False:
                # Position reduced (take profit), keep tracking
                profit_target_hit = True
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                profit_target_hit = False
        
        signals[i] = new_signal
    
    return signals