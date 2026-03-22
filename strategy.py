#!/usr/bin/env python3
"""
Experiment #017: 1d HMA Trend + RSI Pullback + 1w HTF Filter with ATR Trail

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance
of signal quality and trade frequency (20-50 trades/year). Based on proven
pattern from research: HMA crossover + RSI filter + ATR trail achieved
Sharpe +0.879 on SOL. This adapts it for BTC/ETH/SOL with:

1. 1d primary timeframe (proven to work in bear/range markets)
2. 1w HMA as single HTF trend bias (simpler than dual 12h+1d)
3. RSI(14) pullback entry within trend (buy dips in uptrend)
4. ATR(14) trailing stop for risk management
5. Looser RSI thresholds to ensure trade frequency

Why this should work:
- 1d TF naturally filters noise, targets 20-50 trades/year
- Weekly HMA provides strong trend bias without over-filtering
- RSI pullback entries catch continuations (not tops/bottoms)
- ATR trail protects gains while allowing trend to run
- Simpler logic = fewer conditions that can all fail = more trades

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_trend_atr_v1"
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
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    return rsi

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND BIAS (1w) ===
        # Weekly HMA determines overall bias
        htf_bullish = close[i] > hma_1w_21_aligned[i]
        htf_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        # HMA21 > HMA50 = bullish structure
        trend_bullish = (hma_21[i] > hma_50[i]) and (close[i] > hma_21[i])
        trend_bearish = (hma_21[i] < hma_50[i]) and (close[i] < hma_21[i])
        
        # === SMA200 FILTER ===
        # Only long above SMA200, only short below SMA200
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 35-50 in uptrend
        # Short: RSI rallied to 50-65 in downtrend
        rsi_long_pullback = (rsi_14[i] >= 35) and (rsi_14[i] <= 55)
        rsi_short_pullback = (rsi_14[i] >= 45) and (rsi_14[i] <= 65)
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        # Round to discrete levels
        if current_size > 0.27:
            current_size = 0.30
        elif current_size > 0.17:
            current_size = 0.20
        else:
            current_size = 0.15
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: HTF bullish + trend bullish + above SMA200 + RSI pullback
        if htf_bullish and trend_bullish and above_sma200 and rsi_long_pullback:
            new_signal = current_size
        
        # SHORT ENTRY: HTF bearish + trend bearish + below SMA200 + RSI pullback
        elif htf_bearish and trend_bearish and below_sma200 and rsi_short_pullback:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 20 bars (~20 days on 1d), allow weaker entry
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            # Weaker: only need HTF + one trend confirmation
            if htf_bullish and (close[i] > hma_21[i]) and above_sma200:
                new_signal = current_size * 0.7
            elif htf_bearish and (close[i] < hma_21[i]) and below_sma200:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if HMA21 crosses below HMA50
            if position_side > 0 and hma_21[i] < hma_50[i]:
                trend_reversal = True
            # Exit short if HMA21 crosses above HMA50
            if position_side < 0 and hma_21[i] > hma_50[i]:
                trend_reversal = True
        
        # === HTF TREND REVERSAL EXIT ===
        htf_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                htf_reversal = True
            if position_side < 0 and htf_bullish:
                htf_reversal = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or htf_reversal:
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