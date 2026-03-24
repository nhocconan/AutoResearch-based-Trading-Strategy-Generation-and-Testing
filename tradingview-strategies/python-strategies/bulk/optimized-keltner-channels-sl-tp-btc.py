#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Optimized Keltner Channels SL/TP Strategy for BTC"
timeframe = "4h"
leverage = 1

def generate_signals(prices):
    """
    Generates trading signals based on Keltner Channels with SL/TP.
    Args:
        prices (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
    Returns:
        np.array: Array of signals (1.0 for Long, -1.0 for Short, 0.0 for Flat).
    """
    df = prices.copy()
    n = len(df)
    if n == 0:
        return np.array([])
    
    signals = np.zeros(n)
    
    # Strategy Parameters
    length = 9
    mult = 1.0
    atr_length = 19
    sl_pct = 0.20
    tp_pct = 0.203
    
    # Calculate EMA (Basis)
    df['ma'] = df['close'].ewm(span=length, adjust=False).mean()
    
    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=atr_length, adjust=False).mean()
    
    # Calculate Bands
    df['upper'] = df['ma'] + df['atr'] * mult
    df['lower'] = df['ma'] - df['atr'] * mult
    
    # Handle NaNs safely without lookahead (ffill only)
    # Leading NaNs remain NaN, preventing trades during warmup
    df.ffill(inplace=True)
    
    # Extract numpy arrays for performance
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    uppers = df['upper'].values
    lowers = df['lower'].values
    
    # State variables
    position = 0  # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    pending_entry = False
    pending_side = 0
    
    for i in range(n):
        # 1. Execute Pending Entries (Fill at Open of current bar)
        if pending_entry:
            position = pending_side
            entry_price = opens[i]
            pending_entry = False
            pending_side = 0
        
        # 2. Record Signal for current bar (Position held during bar)
        signals[i] = position
        
        # 3. Check Exit Conditions (SL/TP) for current position
        # Checks intrabar high/low of current bar i
        if position == 1:  # Long
            sl_level = entry_price * (1 - sl_pct)
            tp_level = entry_price * (1 + tp_pct)
            if lows[i] <= sl_level or highs[i] >= tp_level:
                position = 0
                entry_price = 0.0
        elif position == -1:  # Short
            sl_level = entry_price * (1 + sl_pct)
            tp_level = entry_price * (1 - tp_pct)
            if highs[i] >= sl_level or lows[i] <= tp_level:
                position = 0
                entry_price = 0.0
        
        # 4. Check Entry Conditions (if flat after exit check)
        # Entry triggered by close of bar i, executed at open of bar i+1
        if position == 0:
            # Ensure indicators are valid before trading
            if not np.isnan(uppers[i]) and not np.isnan(lowers[i]):
                if closes[i] > uppers[i]:
                    pending_entry = True
                    pending_side = 1
                elif closes[i] < lowers[i]:
                    pending_entry = True
                    pending_side = -1
    
    return signals